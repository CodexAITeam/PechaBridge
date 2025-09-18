import os
from PIL import Image, ImageDraw, ImageFont

from tibetanDataGenerator.data.augmentation import AugmentationStrategy


class ImageBuilder:
    def __init__(self, image_size=(1024, 1024)):
        self.image = Image.new('RGB', image_size, color='white')  # Leeres Bild standardmäßig
        self.draw = ImageDraw.Draw(self.image)
        self.font = None

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
        max_y = box_y + box_h

        if rotation == 0:
            # Standard horizontal text rendering
            wrapped_text = []
            for line in text.split('\n'):
                while line:
                    for i in range(1, len(line) + 1):
                        if self.draw.textlength(line[:i], font=self.font) > box_w:
                            break
                    else:
                        i = len(line)

                    wrapped_text.append(line[:i])
                    line = line[i:]

            y_offset = 0
            for line in wrapped_text:
                left, top, right, bottom = self.font.getbbox(line)
                line_height = bottom - top
                if box_y + y_offset + line_height > max_y:
                    break
                self.draw.text((box_x, box_y + y_offset), line, font=self.font, fill=(0, 0, 0))
                y_offset += line_height
        
        elif rotation == 90:
            # Vertical text rendering (90 degrees clockwise)
            # Create a temporary image for the rotated text
            temp_img = Image.new('RGBA', (box_h, box_w), (255, 255, 255, 0))
            temp_draw = ImageDraw.Draw(temp_img)
            
            # Render text on temporary image
            wrapped_text = []
            for line in text.split('\n'):
                while line:
                    for i in range(1, len(line) + 1):
                        if temp_draw.textlength(line[:i], font=self.font) > box_h:
                            break
                    else:
                        i = len(line)
                    wrapped_text.append(line[:i])
                    line = line[i:]
            
            y_offset = 0
            for line in wrapped_text:
                left, top, right, bottom = self.font.getbbox(line)
                line_height = bottom - top
                if y_offset + line_height > box_w:
                    break
                temp_draw.text((0, y_offset), line, font=self.font, fill=(0, 0, 0))
                y_offset += line_height
            
            # Rotate the temporary image and paste it
            rotated = temp_img.rotate(-90, expand=True)
            self.image.paste(rotated, (box_x, box_y), rotated)
        
        else:
            # For other rotations, fall back to standard rendering
            print(f"Warning: Rotation {rotation}° not fully supported, using 0°")
            return self.add_text(text, position, box_size, rotation=0)

        return self

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