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
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_tokens_per_line = max_tokens_per_line
        self.font_size = font_size
        self.padding = padding
        self.font = get_font(font_size)
        self.small_font = get_font(font_size - 4)

        # Get mask token id
        if hasattr(tokenizer, 'mask_token_id'):
            self.mask_token_id = tokenizer.mask_token_id
        else:
            self.mask_token_id = model.config.mask_token_id

    def generate_with_history(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        num_steps: int = 64,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> List[Tuple[List[str], List[float]]]:
        """
        Generate text and capture token/confidence history at each step.

        Returns:
            List of (tokens, confidences) tuples for each refinement step
        """
        self.model.eval()
        history = []

        # Encode prompt
        if hasattr(self.tokenizer, 'encode'):
            prompt_ids = self.tokenizer.encode(prompt)
        else:
            prompt_ids = self.tokenizer(prompt)["input_ids"]

        prompt_ids = torch.tensor([prompt_ids], device=self.device)
        prompt_len = prompt_ids.size(1)

        # Initialize canvas with prompt + masks
        canvas = torch.cat([
            prompt_ids,
            torch.full((1, max_new_tokens), self.mask_token_id, device=self.device)
        ], dim=1)

        # Track which positions are prompt (don't modify)
        prompt_mask = torch.zeros(canvas.size(1), dtype=torch.bool, device=self.device)
        prompt_mask[:prompt_len] = True

        # Capture initial state
        tokens, confidences = self._decode_canvas(canvas[0], prompt_len)
        history.append((tokens, confidences))

        with torch.no_grad():
            for step in range(num_steps):
                # Get mask positions (non-prompt, still masked)
                mask_positions = (canvas == self.mask_token_id) & ~prompt_mask.unsqueeze(0)
                has_masks = mask_positions.any()

                # Forward pass with refinement (always run for continued refinement)
                outputs = self.model(
                    canvas,
                    num_refine_steps=4,
                    return_confidence=True,
                )

                logits = outputs["logits"]
                confidence = outputs.get("confidence", None)

                # Get confidence scores
                if confidence is not None:
                    conf_scores = confidence
                else:
                    probs = F.softmax(logits, dim=-1)
                    conf_scores = probs.max(dim=-1).values

                # Only fill masks if there are any left
                if has_masks:
                    # Apply temperature and top-k
                    scaled_logits = logits / temperature
                    if top_k > 0:
                        indices_to_remove = scaled_logits < torch.topk(scaled_logits, top_k, dim=-1).values[..., -1:]
                        scaled_logits[indices_to_remove] = float('-inf')

                    probs = F.softmax(scaled_logits, dim=-1)

                    # Sample predictions
                    if temperature > 0:
                        predictions = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(probs.size()[:-1])
                    else:
                        predictions = probs.argmax(dim=-1)

                    # Select top-k most confident masked positions to fill
                    # Distribute filling across all steps evenly
                    total_masks = max_new_tokens
                    masks_remaining = mask_positions.sum().item()
                    steps_remaining = num_steps - step
                    # Calculate how many to fill this step to finish exactly at num_steps
                    target_remaining = total_masks * (steps_remaining - 1) / num_steps
                    num_to_fill = max(1, int(masks_remaining - target_remaining))
                    num_to_fill = min(num_to_fill, masks_remaining)  # Don't overfill

                    masked_conf = conf_scores.clone()
                    masked_conf[~mask_positions] = -float('inf')

                    _, top_indices = torch.topk(masked_conf.view(-1), min(num_to_fill, mask_positions.sum().item()))

                    # Fill selected positions
                    canvas_flat = canvas.view(-1)
                    predictions_flat = predictions.view(-1)
                    canvas_flat[top_indices] = predictions_flat[top_indices]
                    canvas = canvas_flat.view(1, -1)

                # Capture state (always, even after all masks filled - shows refinement)
                tokens, confidences = self._decode_canvas(canvas[0], prompt_len, conf_scores[0])
                history.append((tokens, confidences))

        return history

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
        step: int,
        total_steps: int,
        width: int = 800,
        height: int = None,
    ) -> Image.Image:
        """Render a single frame showing tokens with confidence colors."""
        # Calculate layout
        char_width = self.font_size * 0.6
        token_spacing = 5
        line_height = self.font_size + 15

        # Wrap tokens into lines
        lines = []
        current_line = []
        current_width = 0
        max_width = width - 2 * self.padding

        for i, (token, conf) in enumerate(zip(tokens, confidences)):
            token_width = len(token) * char_width + token_spacing

            if current_width + token_width > max_width and current_line:
                lines.append(current_line)
                current_line = []
                current_width = 0

            current_line.append((token, conf))
            current_width += token_width

        if current_line:
            lines.append(current_line)

        # Calculate image height (use fixed if provided)
        header_height = 80
        prompt_height = 40
        content_height = len(lines) * line_height + 20
        footer_height = 40
        if height is None:
            height = header_height + prompt_height + content_height + footer_height

        # Create image
        img = Image.new('RGB', (width, height), color=(30, 30, 40))
        draw = ImageDraw.Draw(img)

        # Draw header
        title = "✨ Shimmer/LIRA Generation"
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

        step_text = f"Step {step}/{total_steps - 1}"
        draw.text((width - self.padding - 100, 45), step_text, fill=(150, 150, 160), font=self.small_font)

        # Draw prompt
        y = header_height
        draw.text((self.padding, y), f"Prompt: \"{prompt}\"", fill=(180, 180, 190), font=self.small_font)
        y += prompt_height

        # Draw tokens
        for line in lines:
            x = self.padding
            for token, conf in line:
                color = confidence_to_color(conf)

                # Draw token background
                token_width = int(len(token) * char_width)
                draw.rectangle(
                    [x - 2, y - 2, x + token_width + 2, y + self.font_size + 2],
                    fill=(50, 50, 60),
                    outline=color,
                    width=1
                )

                # Draw token text
                text_color = color if conf > 0.3 else (200, 100, 100)
                draw.text((x, y), token, fill=text_color, font=self.font)

                x += token_width + token_spacing

            y += line_height

        # Draw legend
        legend_y = height - footer_height + 10
        draw.text((self.padding, legend_y), "Confidence:", fill=(150, 150, 160), font=self.small_font)

        # Draw gradient legend
        legend_x = self.padding + 80
        for i in range(100):
            conf = i / 100
            color = confidence_to_color(conf)
            draw.rectangle([legend_x + i * 2, legend_y, legend_x + i * 2 + 2, legend_y + 15], fill=color)

        draw.text((legend_x - 5, legend_y + 2), "0", fill=(150, 150, 160), font=self.small_font)
        draw.text((legend_x + 200, legend_y + 2), "1", fill=(150, 150, 160), font=self.small_font)

        return img

    def create_gif(
        self,
        prompt: str,
        output_path: str = "generation.gif",
        max_new_tokens: int = 64,
        num_steps: int = 64,
        temperature: float = 0.8,
        frame_duration: float = 0.15,
        width: int = 800,
        hold_final_frames: int = 10,
    ) -> str:
        """
        Generate text and create animated GIF of the process.

        Args:
            prompt: Starting text
            output_path: Where to save the GIF
            max_new_tokens: Number of tokens to generate
            num_steps: Refinement steps
            temperature: Sampling temperature
            frame_duration: Seconds per frame
            width: Image width
            hold_final_frames: Extra frames to hold on final result

        Returns:
            Path to saved GIF
        """
        if not HAS_IMAGEIO:
            raise ImportError("imageio required for GIF creation. Install with: pip install imageio")

        print(f"Generating with {num_steps} steps...")
        history = self.generate_with_history(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            num_steps=num_steps,
            temperature=temperature,
        )

        # Calculate fixed height based on max tokens
        char_width = self.font_size * 0.6
        token_spacing = 5
        line_height = self.font_size + 15
        tokens_per_line = int((width - 2 * self.padding) / (6 * char_width + token_spacing))
        num_lines = (max_new_tokens + tokens_per_line - 1) // tokens_per_line
        fixed_height = 80 + 40 + num_lines * line_height + 40 + 40  # header + prompt + content + footer + padding

        print(f"Rendering {len(history)} frames...")
        frames = []
        for step, (tokens, confidences) in enumerate(history):
            frame = self.render_frame(
                prompt=prompt,
                tokens=tokens,
                confidences=confidences,
                step=step,
                total_steps=len(history),
                width=width,
                height=fixed_height,
            )
            frames.append(np.array(frame))

        # Hold final frame
        for _ in range(hold_final_frames):
            frames.append(frames[-1])

        print(f"Saving GIF to {output_path}...")
        # duration is in seconds for imageio v3+, convert to ms for older versions
        # Use fps instead for better compatibility
        fps = 1.0 / frame_duration
        imageio.mimsave(output_path, frames, fps=fps, loop=0)

        print(f"✨ GIF saved: {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Visualize LIRA generation process")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, help="Path to tokenizer model")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Generation prompt")
    parser.add_argument("--output", type=str, default="generation.gif", help="Output GIF path")
    parser.add_argument("--max_tokens", type=int, default=64, help="Max tokens to generate")
    parser.add_argument("--num_steps", type=int, default=64, help="Refinement steps")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--width", type=int, default=900, help="Image width")
    parser.add_argument("--frame_duration", type=float, default=0.12, help="Seconds per frame")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

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

    # Create visualizer
    visualizer = GenerationVisualizer(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        font_size=18,
    )

    # Generate GIF
    visualizer.create_gif(
        prompt=args.prompt,
        output_path=args.output,
        max_new_tokens=args.max_tokens,
        num_steps=args.num_steps,
        temperature=args.temperature,
        width=args.width,
        frame_duration=args.frame_duration,
    )


if __name__ == "__main__":
    main()
