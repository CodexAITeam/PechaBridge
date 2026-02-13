import os
import time
from PIL import Image, ImageDraw, ImageFont, ImageChops

from tibetanDataGenerator.data.augmentation import AugmentationStrategy


class ImageBuilder:
    def __init__(self, image_size=(1024, 1024)):
        self.image = Image.new('RGB', image_size, color='white')  # Leeres Bild standardmäßig
        self.draw = ImageDraw.Draw(self.image)
        self.font = None
        self._last_text_drawn = False
        self._last_text_bbox = None  # (x, y, w, h) in absolute image coordinates
        self._last_rendered_lines = []
        self._last_rendered_text = ""

    def set_background(self, background_path):
        """
        Setze ein Hintergrundbild für das Bild.
        """
        if not os.path.exists(background_path):
            raise FileNotFoundError(f"Background image {background_path} not found.")
        self.image = Image.open(background_path).resize(self.image.size, Image.Resampling.LANCZOS)
        self.draw = ImageDraw.Draw(self.image)
        return self

    def set_font(self, font_path, font_size=24):
        """
        Lade eine Schriftart für das Rendern von Text.
        """
        try:
            self.font = ImageFont.truetype(font_path, font_size)
        except IOError:
            self.font = ImageFont.load_default()
            print("Warning: Default font used.")
        return self

    def add_text(self, text, position, box_size, rotation=0):
        """
        Fügt Text auf dem Bild an einer bestimmten Position mit automatischer Begrenzung hinzu.
        
        Args:
            text: Text to render
            position: (x, y) position
            box_size: (width, height) of text box
            rotation: Rotation angle in degrees (0, 90, 180, 270)
        """
        if not self.font:
            raise ValueError("Font not set. Use set_font() before adding text.")

        box_x, box_y = position
        box_w, box_h = box_size
        if box_w <= 0 or box_h <= 0:
            self._last_text_drawn = False
            self._last_text_bbox = None
            self._last_rendered_lines = []
            self._last_rendered_text = ""
            return self
        if not text or not str(text).strip():
            self._last_text_drawn = False
            self._last_text_bbox = None
            self._last_rendered_lines = []
            self._last_rendered_text = ""
            return self
        max_y = box_y + box_h
        drew_any = False
        drawn_lines = []
        before = self.image.copy()

        if rotation == 0:
            # Standard horizontal text rendering
            wrapped_text = self._wrap_text_lines(self.draw, text, box_w)

            y_offset = 0
            for line in wrapped_text:
                line_height = self._safe_line_height(line)
                if box_y + y_offset + line_height > max_y:
                    break
                self.draw.text((box_x, box_y + y_offset), line, font=self.font, fill=(0, 0, 0))
                y_offset += line_height
                drew_any = True
                drawn_lines.append(line)
        
        elif rotation == 90:
            # Vertical text rendering (90 degrees clockwise)
            # Create a temporary image for the rotated text
            temp_img = Image.new('RGBA', (box_h, box_w), (255, 255, 255, 0))
            temp_draw = ImageDraw.Draw(temp_img)
            
            # Render text on temporary image
            wrapped_text = self._wrap_text_lines(temp_draw, text, box_h)
            
            y_offset = 0
            for line in wrapped_text:
                line_height = self._safe_line_height(line)
                if y_offset + line_height > box_w:
                    break
                temp_draw.text((0, y_offset), line, font=self.font, fill=(0, 0, 0))
                y_offset += line_height
                drew_any = True
                drawn_lines.append(line)
            
            # Rotate the temporary image and paste it
            if drew_any:
                rotated = temp_img.rotate(-90, expand=True)
                self.image.paste(rotated, (box_x, box_y), rotated)
        
        else:
            # For other rotations, fall back to standard rendering
            print(f"Warning: Rotation {rotation}° not fully supported, using 0°")
            return self.add_text(text, position, box_size, rotation=0)

        # If a rotated rendering produced nothing, fall back to horizontal text
        # so we never create an empty labeled region.
        if not drew_any and rotation != 0:
            return self.add_text(text, position, box_size, rotation=0)

        self._last_text_drawn = drew_any
        if drew_any:
            diff = ImageChops.difference(self.image, before)
            bbox = diff.getbbox()
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                self._last_text_bbox = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))
                self._last_rendered_lines = list(drawn_lines)
                self._last_rendered_text = "\n".join(drawn_lines)
            else:
                self._last_text_bbox = None
                self._last_text_drawn = False
                self._last_rendered_lines = []
                self._last_rendered_text = ""
        else:
            self._last_text_bbox = None
            self._last_rendered_lines = []
            self._last_rendered_text = ""
        return self

    def last_text_drawn(self):
        return bool(self._last_text_drawn)

    def last_text_bbox(self):
        return self._last_text_bbox

    def last_rendered_lines(self):
        return list(self._last_rendered_lines)

    def last_rendered_text(self):
        return self._last_rendered_text

    def _wrap_text_lines(self, draw_ctx, text, max_width, timeout_seconds=1.5, max_total_lines=300):
        """
        Robust text wrapping with hard limits to avoid pathological hangs in PIL font shaping.
        """
        start = time.time()
        wrapped = []

        for raw_line in text.split('\n'):
            line = raw_line
            chunk_guard = 0
            while line and len(wrapped) < max_total_lines:
                if time.time() - start > timeout_seconds:
                    # Stop wrapping if it becomes too expensive.
                    return wrapped
                if chunk_guard > 10000:
                    # Safety stop for malformed/unexpected input.
                    return wrapped

                take = self._max_prefix_that_fits(draw_ctx, line, max_width)
                if take <= 0:
                    take = 1
                wrapped.append(line[:take])
                line = line[take:]
                chunk_guard += 1

        return wrapped

    def _max_prefix_that_fits(self, draw_ctx, line, max_width):
        """
        Find longest prefix that fits max_width via binary search.
        This avoids O(n^2) behavior from character-by-character scanning.
        """
        if not line:
            return 0

        lo, hi = 1, len(line)
        best = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            snippet = line[:mid]
            width = self._safe_textlength(draw_ctx, snippet)
            if width <= max_width:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1

        # Ensure forward progress even if first glyph is wider than max_width.
        return best if best > 0 else 1

    def _safe_textlength(self, draw_ctx, text):
        try:
            return draw_ctx.textlength(text, font=self.font)
        except Exception:
            # Fallback heuristic if PIL shaping fails on a specific glyph sequence.
            return float(len(text) * 10)

    def _safe_line_height(self, line):
        sample = line if line else "A"
        try:
            left, top, right, bottom = self.font.getbbox(sample)
            h = bottom - top
            if h > 0:
                return h
        except Exception:
            pass

        try:
            asc, desc = self.font.getmetrics()
            if asc + desc > 0:
                return asc + desc
        except Exception:
            pass

        return 16

    def add_bounding_box(self, position, size, color=(255, 0, 0)):
        """
        Zeichne eine Bounding Box auf dem Bild.
        """
        x, y = position
        w, h = size
        self.draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
        return self

    def apply_augmentation(self, augmentation_strategy):
        """
        Apply an augmentation strategy to the current image.

        :param augmentation_strategy: An instance of AugmentationStrategy
        :return: self for method chaining
        """
        if not isinstance(augmentation_strategy, AugmentationStrategy):
            raise ValueError("augmentation_strategy must be an instance of AugmentationStrategy")

        self.image = augmentation_strategy.apply(self.image)
        self.draw = ImageDraw.Draw(self.image)
        return self


    def save(self, output_path):
        """
        Speichert das fertige Bild.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.image.save(output_path)
        return self

    def show(self):
        """
        Zeigt das Bild zur Vorschau an.
        """
        self.image.show()
        return self
