import os
import textwrap
from PIL import Image, ImageDraw, ImageFont


class Path:
    def __init__(self):
        self.screenshots = []
        self.base_filename = "agent_path"
        self.path_history_dir = "path-history"
        os.makedirs(self.path_history_dir, exist_ok=True)

    def create_text_image(self, text, width=800, height=100, font_size=100):
        font = ImageFont.load_default()
        image = Image.new("RGB", (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        lines = textwrap.wrap(text, width=80)
        y_text = 10
        for line in lines:
            draw.text((10, y_text), line, fill="black", font=font)
            y_text += font_size + 5
        return image

    def add_screenshot(self, image, is_initial=False, is_final=False):
        if is_initial or is_final:
            self.screenshots.append(image)
        else:
            self.screenshots.append(Image.open(image))
        self.update_agent_path_image(image, is_initial, is_final)

    def update_agent_path_image(self, new_image, is_initial=False, is_final=False):
        if isinstance(new_image, str):
            img = Image.open(new_image)
        elif isinstance(new_image, Image.Image):
            img = new_image
        else:
            raise ValueError(
                "The new_image parameter must be a file path or a PIL Image object."
            )

        if is_initial or is_final:
            self.screenshots.insert(0 if is_initial else len(self.screenshots), img)
        else:
            self.screenshots.append(img)
        filename = os.path.join(self.path_history_dir, f"{self.base_filename}.png")

        cols = 3
        rows = (len(self.screenshots) + cols - 1) // cols
        max_width, max_height = self.get_max_dimensions(self.screenshots)

        grid_image = Image.new(
            "RGB", (cols * max_width, rows * max_height), color=(255, 255, 255)
        )
        for i, img in enumerate(self.screenshots):
            x = (i % cols) * max_width
            y = (i // cols) * max_height
            grid_image.paste(img, (x, y))
        grid_image.save(filename)

    def get_max_dimensions(self, screenshots):
        widths, heights = zip(
            *[
                (
                    (img.width, img.height)
                    if isinstance(img, Image.Image)
                    else Image.open(img).size
                )
                for img in screenshots
            ]
        )
        return max(widths), max(heights)