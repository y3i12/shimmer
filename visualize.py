#!/usr/bin/env python3
"""
Shimmer/LIRA Generation Visualization

Creates animated GIFs showing how tokens evolve through refinement iterations.
Tokens are color-coded by confidence: red (uncertain) → green (confident).
"""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import List, Tuple, Optional
import io

# Try to import imageio for GIF creation
try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("Warning: imageio not installed. Install with: pip install imageio")

from lira import LatentCanvasModel, LatentCanvasConfig
from lira.tokenizer import ShimmerTokenizer


class SemanticCritic:
    """
    Uses a pretrained LM (GPT-2) to score semantic coherence.
    High surprise = semantically odd = candidate for revision.
    """

    def __init__(self, model_name: str = "gpt2", device: str = "cuda"):
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        print(f"Loading semantic critic: {model_name}...")
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.device = device
        self.model.eval()
        print(f"  Critic loaded: {sum(p.numel() for p in self.model.parameters()):,} params")

    def get_per_token_surprise(
        self,
        text: str,
        return_tokens: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[str]]]:
        """
        Compute per-token surprise (negative log probability).
        Higher surprise = less likely = semantically odd.

        Returns:
            surprise: Tensor of shape [num_tokens] with surprise values
            tokens: Optional list of token strings
        """
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # [1, seq_len, vocab]

        # Compute log probs for actual tokens
        # Shift: logits[t] predicts token[t+1]
        log_probs = F.log_softmax(logits[:, :-1], dim=-1)  # [1, seq_len-1, vocab]
        target_ids = input_ids[:, 1:]  # [1, seq_len-1]

        # Gather log probs for actual tokens
        token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)  # [1, seq_len-1]

        # Surprise = negative log prob (higher = more surprising)
        surprise = -token_log_probs[0]  # [seq_len-1]

        # Prepend 0 for first token (no prediction for it)
        surprise = torch.cat([torch.zeros(1, device=self.device), surprise])

        if return_tokens:
            tokens = [self.tokenizer.decode([tid]) for tid in input_ids[0].tolist()]
            return surprise, tokens

        return surprise, None

    def score_canvas(
        self,
        canvas_text: str,
        prompt_len_chars: int,
    ) -> Tuple[List[float], float]:
        """
        Score a generated canvas, returning per-position surprise.

        Returns:
            per_token_surprise: List of surprise values for generated region
            mean_surprise: Average surprise for the generation
        """
        surprise, tokens = self.get_per_token_surprise(canvas_text, return_tokens=True)

        # We need to map GPT-2 tokens back to approximate character positions
        # This is approximate since tokenizations may differ
        char_pos = 0
        token_char_positions = []
        for tok in tokens:
            token_char_positions.append(char_pos)
            char_pos += len(tok)

        # Find tokens that fall in generated region
        gen_surprises = []
        for i, pos in enumerate(token_char_positions):
            if pos >= prompt_len_chars:
                gen_surprises.append(surprise[i].item())

        mean_surprise = sum(gen_surprises) / len(gen_surprises) if gen_surprises else 0.0

        return gen_surprises, mean_surprise


def confidence_to_color(confidence: float) -> Tuple[int, int, int]:
    """Convert confidence (0-1) to RGB color (red → yellow → green)."""
    if confidence < 0.5:
        # Red to Yellow
        r = 255
        g = int(255 * (confidence * 2))
        b = 50
    else:
        # Yellow to Green
        r = int(255 * (2 - confidence * 2))
        g = 255
        b = 50
    return (r, g, b)


def get_font(size: int = 16):
    """Try to load a monospace font, fall back to default."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "C:/Windows/Fonts/consola.ttf",
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


class GenerationVisualizer:
    """Captures and visualizes LIRA generation process."""

    def __init__(
        self,
        model: LatentCanvasModel,
        tokenizer,
        device: str = "cuda",
        max_tokens_per_line: int = 12,
        font_size: int = 18,
        padding: int = 20,
        semantic_critic: Optional[SemanticCritic] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_tokens_per_line = max_tokens_per_line
        self.font_size = font_size
        self.padding = padding
        self.font = get_font(font_size)
        self.small_font = get_font(font_size - 4)
        self.semantic_critic = semantic_critic

        # Get mask token id
        if hasattr(tokenizer, 'mask_token_id'):
            self.mask_token_id = tokenizer.mask_token_id
        else:
            self.mask_token_id = model.config.mask_token_id

    def generate_with_history(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        num_steps: int = None,
        max_passes: int = None,
        revision_threshold: float = 0.7,
        coherence_threshold: float = 0.7,
        use_coherence: bool = True,
        temperature: float = 0.8,
        use_semantic_critic: bool = False,
        surprise_threshold: float = 5.0,
        **kwargs,
    ) -> Tuple[List[Tuple[List[str], List[float], List[bool]]], dict]:
        """
        Generate text using topk approach, then optionally revise.

        Phase 1: Fill masks gradually (most confident first) - stable
        Phase 2: Revise low-confidence OR low-coherence tokens (if max_passes > num_steps)

        Returns:
            history: List of (tokens, confidences, revised_flags) tuples
            stats: Generation statistics
        """
        self.model.eval()

        # Encode prompt (same pattern as generate.py)
        from lira.tokenizer import ShimmerTokenizer
        if isinstance(self.tokenizer, ShimmerTokenizer):
            prompt_ids = self.tokenizer.encode(prompt, add_bos=False, add_eos=False)
        elif hasattr(self.tokenizer, 'encode'):
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        else:
            prompt_ids = self.tokenizer(prompt)["input_ids"]

        prompt_ids = torch.tensor([prompt_ids], device=self.device)
        prompt_len = prompt_ids.size(1)
        mask_token = self.mask_token_id
        gen_len = max_new_tokens

        # Initialize canvas: prompt + masks
        canvas = torch.cat([
            prompt_ids,
            torch.full((1, gen_len), mask_token, device=self.device, dtype=torch.long)
        ], dim=1)

        # Default: one token per step for smooth animation
        if num_steps is None:
            num_steps = gen_len
        if max_passes is None:
            max_passes = num_steps  # No revision by default
        tokens_per_step = max(1, gen_len // num_steps)

        history = []
        stats = {
            "total_passes": 0,
            "total_revisions": 0,
            "masks_filled": 0,
            "stopped_by": None,
        }

        # Capture initial state (all masks)
        tokens, confidences = self._decode_canvas(canvas[0], prompt_len, None)
        history.append((tokens, confidences, [False] * len(tokens)))

        with torch.no_grad():
            # Phase 1: Fill masks using topk (stable)
            for step in range(num_steps):
                stats["total_passes"] += 1

                # Forward pass
                result = self.model(canvas, return_confidence=True)
                logits = result["logits"]
                model_confidence = result.get("confidence", None)

                # Find remaining mask positions
                mask_positions = (canvas[0] == mask_token).nonzero(as_tuple=True)[0]
                if len(mask_positions) == 0:
                    break  # All masks filled, move to phase 2

                # Get logits and confidence for mask positions only
                gen_logits = logits[0, mask_positions] / temperature
                probs = torch.softmax(gen_logits, dim=-1)
                confidence_scores, _ = probs.max(dim=-1)

                # Select top-k most confident masks to fill
                k = min(tokens_per_step, len(mask_positions))
                top_k_conf, top_k_idx = confidence_scores.topk(k)

                # Fill the selected masks
                filled_positions = set()
                for idx in top_k_idx:
                    pos = mask_positions[idx]
                    sampled_token = torch.multinomial(probs[idx], num_samples=1).item()
                    canvas[0, pos] = sampled_token
                    stats["masks_filled"] += 1
                    filled_positions.add((pos - prompt_len).item())

                # Capture state
                if model_confidence is not None:
                    tokens, confidences = self._decode_canvas(canvas[0], prompt_len, model_confidence[0])
                else:
                    tokens, confidences = self._decode_canvas(canvas[0], prompt_len, None)
                revised_flags = [(i in filled_positions) for i in range(len(tokens))]
                history.append((tokens, confidences, revised_flags))

            # Phase 2: Stable revision passes (if max_passes > steps used)
            # Key insight: Revising ALL low-conf tokens causes cascade destabilization
            # Solution: 1. Lock high-confidence tokens  2. Revise only 1 token per pass

            remaining_passes = max_passes - stats["total_passes"]
            locked_positions = set()  # Positions that are "stable" and won't be revised
            lock_threshold = 0.85  # Once confidence exceeds this, lock the token

            # Track semantic critic usage
            if use_semantic_critic and self.semantic_critic is not None:
                stats["using_semantic_critic"] = True
            else:
                stats["using_semantic_critic"] = False

            for pass_num in range(remaining_passes):
                stats["total_passes"] += 1

                # Forward pass
                result = self.model(canvas, return_confidence=True)
                logits = result["logits"]
                confidence = result.get("confidence", None)

                if confidence is None:
                    probs = F.softmax(logits, dim=-1)
                    confidence = probs.max(dim=-1).values

                # Lock high-confidence tokens (they become stable anchors)
                for pos in range(prompt_len, canvas.size(1)):
                    pos_confidence = confidence[0, pos].item()
                    if pos_confidence >= lock_threshold:
                        locked_positions.add(pos)

                # === SEMANTIC CRITIC: Find revision candidates ===
                if use_semantic_critic and self.semantic_critic is not None:
                    # Decode current canvas to text
                    canvas_tokens = []
                    for tid in canvas[0].tolist():
                        if tid == self.mask_token_id:
                            canvas_tokens.append("[MASK]")
                        else:
                            canvas_tokens.append(self.tokenizer.decode([tid]))
                    full_text = "".join(canvas_tokens)

                    # Get per-token surprise from GPT-2
                    surprise, gpt2_tokens = self.semantic_critic.get_per_token_surprise(
                        full_text, return_tokens=True
                    )

                    # Map surprise back to LIRA token positions (approximate)
                    # Build character position map for LIRA tokens
                    lira_char_positions = []
                    char_pos = 0
                    for tok in canvas_tokens:
                        lira_char_positions.append(char_pos)
                        char_pos += len(tok)

                    # Build character position map for GPT-2 tokens
                    gpt2_char_positions = []
                    char_pos = 0
                    for tok in gpt2_tokens:
                        gpt2_char_positions.append(char_pos)
                        char_pos += len(tok)

                    # For each LIRA position in gen region, find closest GPT-2 surprise
                    lira_surprises = []
                    for lira_pos in range(prompt_len, canvas.size(1)):
                        lira_char = lira_char_positions[lira_pos]
                        # Find GPT-2 token that covers this character position
                        best_gpt2_idx = 0
                        for g_idx, g_char in enumerate(gpt2_char_positions):
                            if g_char <= lira_char:
                                best_gpt2_idx = g_idx
                            else:
                                break
                        if best_gpt2_idx < len(surprise):
                            lira_surprises.append(surprise[best_gpt2_idx].item())
                        else:
                            lira_surprises.append(0.0)

                    # Find candidates: unlocked AND high surprise (semantically odd)
                    candidates = []
                    for i, pos in enumerate(range(prompt_len, canvas.size(1))):
                        if pos in locked_positions:
                            continue
                        pos_surprise = lira_surprises[i]
                        if pos_surprise > surprise_threshold:
                            # Use negative surprise as "confidence" for sorting (higher surprise = lower score)
                            candidates.append((pos, -pos_surprise))
                else:
                    # Find revision candidates using LIRA confidence AND coherence
                    # Check if model has coherence head
                    has_coherence = hasattr(self.model, 'coherence_head') and self.model.coherence_head is not None
                    coherence = None
                    if has_coherence and use_coherence:
                        coherence = self.model.get_coherence(result.get("latent", None))

                    candidates = []
                    for pos in range(prompt_len, canvas.size(1)):
                        if pos in locked_positions:
                            continue
                        pos_confidence = confidence[0, pos].item()

                        # Check coherence if available
                        pos_coherence = 1.0  # Default: assume coherent
                        if coherence is not None:
                            pos_coherence = coherence[0, pos].item()

                        # Candidate if low confidence OR low coherence
                        is_low_conf = pos_confidence < revision_threshold
                        is_low_coh = pos_coherence < coherence_threshold

                        if is_low_conf or is_low_coh:
                            # Score: average of both (lower = worse = higher priority to revise)
                            score = (pos_confidence + pos_coherence) / 2.0
                            candidates.append((pos, score))

                # No candidates = stable, we're done
                if not candidates:
                    stats["stopped_by"] = "all_stable"
                    # Still capture final state
                    tokens, confidences = self._decode_canvas(canvas[0], prompt_len, confidence[0])
                    history.append((tokens, confidences, [False] * len(tokens)))
                    break

                # Sort by confidence (lowest first) and pick only the worst ONE
                candidates.sort(key=lambda x: x[1])
                worst_pos, worst_conf = candidates[0]

                # Sample prediction for this position
                if temperature > 0:
                    pos_probs = F.softmax(logits[0, worst_pos] / temperature, dim=-1)
                    new_token = torch.multinomial(pos_probs, num_samples=1).item()
                else:
                    new_token = logits[0, worst_pos].argmax().item()

                current_token = canvas[0, worst_pos].item()

                # Revise only if prediction is different
                revised_positions = set()
                if new_token != current_token:
                    canvas[0, worst_pos] = new_token
                    stats["total_revisions"] += 1
                    revised_positions.add(worst_pos - prompt_len)

                # Capture state
                tokens, confidences = self._decode_canvas(canvas[0], prompt_len, confidence[0])
                revised_flags = [(i in revised_positions) for i in range(len(tokens))]
                history.append((tokens, confidences, revised_flags))

                # If we revised but the new token is still low-conf, it might cascade
                # The locking mechanism will prevent stable neighbors from being affected

        if stats["stopped_by"] is None:
            stats["stopped_by"] = "max_passes"

        return history, stats

    def _decode_canvas(
        self,
        canvas: torch.Tensor,
        prompt_len: int,
        confidence: Optional[torch.Tensor] = None,
    ) -> Tuple[List[str], List[float]]:
        """Decode canvas to tokens and get confidences."""
        tokens = []
        confidences = []

        for i, token_id in enumerate(canvas[prompt_len:].tolist()):
            if token_id == self.mask_token_id:
                tokens.append("[MASK]")
                confidences.append(0.0)
            else:
                if hasattr(self.tokenizer, 'decode'):
                    token_str = self.tokenizer.decode([token_id])
                else:
                    token_str = self.tokenizer.decode([token_id])
                tokens.append(token_str)

                if confidence is not None:
                    conf = confidence[prompt_len + i].item()
                else:
                    conf = 1.0
                confidences.append(conf)

        return tokens, confidences

    def render_frame(
        self,
        prompt: str,
        tokens: List[str],
        confidences: List[float],
        revised_flags: List[bool],
        step: int,
        total_steps: int,
        stats: dict,
        width: int = 800,
        height: int = None,
    ) -> Image.Image:
        """Render a single frame showing tokens with confidence colors and revision indicators."""
        # Calculate layout
        char_width = self.font_size * 0.6
        token_spacing = 5
        line_height = self.font_size + 15

        # Wrap tokens into lines
        lines = []
        current_line = []
        current_width = 0
        max_width = width - 2 * self.padding

        for i, (token, conf, revised) in enumerate(zip(tokens, confidences, revised_flags)):
            token_width = len(token) * char_width + token_spacing

            if current_width + token_width > max_width and current_line:
                lines.append(current_line)
                current_line = []
                current_width = 0

            current_line.append((token, conf, revised))
            current_width += token_width

        if current_line:
            lines.append(current_line)

        # Calculate image height (use fixed if provided)
        header_height = 80
        prompt_height = 40
        content_height = len(lines) * line_height + 20
        footer_height = 60  # Increased for stats
        if height is None:
            height = header_height + prompt_height + content_height + footer_height

        # Create image
        img = Image.new('RGB', (width, height), color=(30, 30, 40))
        draw = ImageDraw.Draw(img)

        # Draw header
        title = "✨ Shimmer/LIRA Revision Generation"
        draw.text((self.padding, 15), title, fill=(255, 255, 255), font=self.font)

        # Progress bar
        progress = step / max(1, total_steps - 1)
        bar_width = width - 2 * self.padding
        bar_height = 8
        bar_y = 50
        draw.rectangle(
            [self.padding, bar_y, self.padding + bar_width, bar_y + bar_height],
            fill=(60, 60, 70)
        )
        draw.rectangle(
            [self.padding, bar_y, self.padding + int(bar_width * progress), bar_y + bar_height],
            fill=(100, 200, 255)
        )

        step_text = f"Pass {step}/{total_steps - 1}"
        draw.text((width - self.padding - 100, 45), step_text, fill=(150, 150, 160), font=self.small_font)

        # Draw prompt
        y = header_height
        draw.text((self.padding, y), f"Prompt: \"{prompt}\"", fill=(180, 180, 190), font=self.small_font)
        y += prompt_height

        # Draw tokens
        for line in lines:
            x = self.padding
            for token, conf, revised in line:
                color = confidence_to_color(conf)

                # Draw token background - blue border if revised this pass
                token_width = int(len(token) * char_width)
                border_color = (100, 150, 255) if revised else color
                border_width = 2 if revised else 1

                draw.rectangle(
                    [x - 2, y - 2, x + token_width + 2, y + self.font_size + 2],
                    fill=(50, 50, 60),
                    outline=border_color,
                    width=border_width
                )

                # Draw token text
                text_color = color if conf > 0.3 else (200, 100, 100)
                draw.text((x, y), token, fill=text_color, font=self.font)

                x += token_width + token_spacing

            y += line_height

        # Draw legend and stats
        legend_y = height - footer_height + 5
        draw.text((self.padding, legend_y), "Confidence:", fill=(150, 150, 160), font=self.small_font)

        # Draw gradient legend
        legend_x = self.padding + 80
        for i in range(100):
            conf = i / 100
            color = confidence_to_color(conf)
            draw.rectangle([legend_x + i * 2, legend_y, legend_x + i * 2 + 2, legend_y + 15], fill=color)

        draw.text((legend_x - 5, legend_y + 2), "0", fill=(150, 150, 160), font=self.small_font)
        draw.text((legend_x + 200, legend_y + 2), "1", fill=(150, 150, 160), font=self.small_font)

        # Revision indicator in legend
        draw.rectangle([legend_x + 240, legend_y, legend_x + 260, legend_y + 15], outline=(100, 150, 255), width=2)
        draw.text((legend_x + 265, legend_y), "= revised", fill=(100, 150, 255), font=self.small_font)

        # Stats line
        stats_y = legend_y + 20
        stats_text = f"Revisions: {stats.get('total_revisions', 0)} | Filled: {stats.get('masks_filled', 0)}"
        if stats.get('stopped_by'):
            stats_text += f" | Stopped: {stats['stopped_by']}"
        draw.text((self.padding, stats_y), stats_text, fill=(120, 120, 130), font=self.small_font)

        return img

    def create_gif(
        self,
        prompt: str,
        output_path: str = "generation.gif",
        max_new_tokens: int = 64,
        min_passes: int = 1,
        max_passes: int = 100,
        revision_threshold: float = 0.7,
        coherence_threshold: float = 0.7,
        use_coherence: bool = True,
        temperature: float = 0.8,
        auto_stop: bool = True,
        frame_duration: float = 0.15,
        width: int = 800,
        hold_final_frames: int = 30,
        use_semantic_critic: bool = False,
        surprise_threshold: float = 5.0,
    ) -> str:
        """
        Generate text with revision and create animated GIF of the process.

        Args:
            prompt: Starting text
            output_path: Where to save the GIF
            max_new_tokens: Number of tokens to generate
            min_passes: Minimum refinement passes
            max_passes: Maximum refinement passes
            revision_threshold: Confidence threshold for revision
            temperature: Sampling temperature
            auto_stop: Stop when all positions confident
            frame_duration: Seconds per frame
            width: Image width
            hold_final_frames: Extra frames to hold on final result

        Returns:
            Path to saved GIF
        """
        if not HAS_IMAGEIO:
            raise ImportError("imageio required for GIF creation. Install with: pip install imageio")

        # Phase 1: Fill masks (1 token per step)
        # Phase 2: Revision passes (if max_passes > max_new_tokens)
        num_steps = max_new_tokens
        critic_info = " + GPT-2 semantic critic" if use_semantic_critic else ""
        coherence_info = " + coherence head" if use_coherence else ""
        print(f"Generating: topk fill ({num_steps} steps) + revision (up to {max_passes} total passes){critic_info}{coherence_info}...")
        print(f"  Temperature: {temperature}, Revision threshold: {revision_threshold}, Coherence threshold: {coherence_threshold}")
        if use_semantic_critic:
            print(f"  Semantic critic: surprise_threshold={surprise_threshold}")
        history, stats = self.generate_with_history(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            num_steps=num_steps,
            max_passes=max_passes,
            revision_threshold=revision_threshold,
            coherence_threshold=coherence_threshold,
            use_coherence=use_coherence,
            temperature=temperature,
            use_semantic_critic=use_semantic_critic,
            surprise_threshold=surprise_threshold,
        )

        print(f"Generation stats: {stats}")

        # Print text log of each step
        print(f"\n{'='*60}")
        print("STEP-BY-STEP TEXT LOG")
        print(f"{'='*60}")
        for step, (tokens, confidences, revised_flags) in enumerate(history):
            text = " ".join(tokens).replace("[MASK]", "▢")
            avg_conf = sum(confidences) / len(confidences) if confidences else 0
            num_masks = sum(1 for t in tokens if t == "[MASK]")
            num_revised = sum(revised_flags)
            print(f"\nStep {step:3d} | Masks: {num_masks:2d} | Revised: {num_revised:2d} | AvgConf: {avg_conf:.2f}")
            print(f"  Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"\n{'='*60}")
        print(f"FINAL: {''.join(history[-1][0])}")
        print(f"{'='*60}\n")

        # Calculate fixed height based on max tokens
        char_width = self.font_size * 0.6
        token_spacing = 5
        line_height = self.font_size + 15
        tokens_per_line = int((width - 2 * self.padding) / (6 * char_width + token_spacing))
        num_lines = (max_new_tokens + tokens_per_line - 1) // tokens_per_line
        fixed_height = 80 + 40 + num_lines * line_height + 60 + 40  # header + prompt + content + footer + padding

        print(f"Rendering {len(history)} frames...")
        frames = []
        for step, (tokens, confidences, revised_flags) in enumerate(history):
            frame = self.render_frame(
                prompt=prompt,
                tokens=tokens,
                confidences=confidences,
                revised_flags=revised_flags,
                step=step,
                total_steps=len(history),
                stats=stats,
                width=width,
                height=fixed_height,
            )
            frames.append(np.array(frame))

        # Hold final frame
        for _ in range(hold_final_frames):
            frames.append(frames[-1])

        print(f"Saving GIF to {output_path}...")
        fps = 1.0 / frame_duration
        imageio.mimsave(output_path, frames, fps=fps, loop=0)

        print(f"✨ GIF saved: {output_path}")
        print(f"   Passes: {stats['total_passes']}, Revisions: {stats['total_revisions']}, Stopped by: {stats['stopped_by']}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Visualize LIRA revision generation process")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, help="Path to tokenizer model")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Generation prompt")
    parser.add_argument("--output", type=str, default="generation.gif", help="Output GIF path")
    parser.add_argument("--max_tokens", type=int, default=64, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--width", type=int, default=900, help="Image width")
    parser.add_argument("--frame_duration", type=float, default=0.12, help="Seconds per frame")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Revision parameters
    parser.add_argument("--min_passes", type=int, default=1, help="Minimum refinement passes")
    parser.add_argument("--max_passes", type=int, default=100, help="Maximum refinement passes")
    parser.add_argument("--revision_threshold", type=float, default=0.7,
                        help="Confidence threshold for revision (below = can be revised)")
    parser.add_argument("--coherence_threshold", type=float, default=0.7,
                        help="Coherence threshold for revision (below = semantically odd, can be revised)")
    parser.add_argument("--no_coherence", action="store_true",
                        help="Disable coherence-based revision (use confidence only)")
    parser.add_argument("--no_auto_stop", action="store_true",
                        help="Disable auto-stop when all positions confident")

    # Semantic critic parameters
    parser.add_argument("--semantic_critic", action="store_true",
                        help="Use GPT-2 as semantic critic to guide revision")
    parser.add_argument("--surprise_threshold", type=float, default=5.0,
                        help="Surprise threshold for semantic critic (higher = more surprising = revise)")
    parser.add_argument("--critic_model", type=str, default="gpt2",
                        help="Which GPT-2 model to use as critic (gpt2, gpt2-medium, gpt2-large)")

    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Get config from checkpoint or infer
    if "config" in checkpoint:
        config = checkpoint["config"]
        if isinstance(config, dict):
            config = LatentCanvasConfig(**config)
    else:
        # Infer from state dict
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        hidden_size = state_dict["token_embed.weight"].shape[1]
        vocab_size = state_dict["token_embed.weight"].shape[0] - 1
        num_layers = sum(1 for k in state_dict if k.startswith("blocks.") and k.endswith(".norm1.weight"))
        num_heads = state_dict["blocks.0.attn.qkv_proj.weight"].shape[0] // hidden_size // 3

        config = LatentCanvasConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
        )

    # Create model
    model = LatentCanvasModel(config).to(args.device)

    # Load weights
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Load tokenizer
    if args.tokenizer:
        tokenizer = ShimmerTokenizer(model_path=args.tokenizer)
    else:
        # Try to find tokenizer in same directory
        ckpt_dir = Path(args.checkpoint).parent
        tokenizer_files = list(ckpt_dir.glob("*.model")) + list(Path("tokenizers").glob("*.model"))
        if tokenizer_files:
            tokenizer = ShimmerTokenizer(model_path=str(tokenizer_files[0]))
            print(f"Using tokenizer: {tokenizer_files[0]}")
        else:
            from transformers import GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            print("Using GPT-2 tokenizer")

    # Create semantic critic if requested
    semantic_critic = None
    if args.semantic_critic:
        semantic_critic = SemanticCritic(model_name=args.critic_model, device=args.device)

    # Create visualizer
    visualizer = GenerationVisualizer(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        font_size=18,
        semantic_critic=semantic_critic,
    )

    # Generate GIF with revision
    visualizer.create_gif(
        prompt=args.prompt,
        output_path=args.output,
        max_new_tokens=args.max_tokens,
        min_passes=args.min_passes,
        max_passes=args.max_passes,
        revision_threshold=args.revision_threshold,
        coherence_threshold=args.coherence_threshold,
        use_coherence=not args.no_coherence,
        temperature=args.temperature,
        auto_stop=not args.no_auto_stop,
        width=args.width,
        frame_duration=args.frame_duration,
        use_semantic_critic=args.semantic_critic,
        surprise_threshold=args.surprise_threshold,
    )


if __name__ == "__main__":
    main()
