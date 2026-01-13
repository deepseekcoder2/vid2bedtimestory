from __future__ import annotations

from pathlib import Path

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, inch
from reportlab.platypus import Flowable, Image as RLImage, PageBreak, Paragraph, SimpleDocTemplate, Spacer, KeepTogether
from reportlab.pdfgen import canvas

from .models import BookSpec
from .text_markup import to_reportlab_markup


class TechFrame(Flowable):
    """
    Wraps an image in a 'Hot Wheels' style tech frame with chamfered corners.
    """
    def __init__(self, img_path: Path, width: float, height: float):
        super().__init__()
        self.img_path = img_path
        self.width = width
        self.height = height
        self.border_width = 4
        self.chamfer = 15  # Size of the corner cut in points

    def draw(self):
        # Draw the image centered
        self.canv.saveState()
        
        # Draw Image
        # Note: drawImage places (x, y) at bottom-left. 
        # Flowable coordinate system is relative to bottom-left of the flowable box.
        self.canv.drawImage(
            str(self.img_path), 
            0, 
            0, 
            width=self.width, 
            height=self.height,
            preserveAspectRatio=True,
            anchor='c' # Center
        )

        # Draw Tech Border
        # Color: Orange/Yellow gradient approximated as solid Orange for MVP
        self.canv.setStrokeColor(colors.orange)
        self.canv.setLineWidth(self.border_width)
        self.canv.setFillColor(colors.transparent)

        # Calculate chamfered path
        w = self.width
        h = self.height
        c = self.chamfer

        p = self.canv.beginPath()
        # Start top-left chamfer
        p.moveTo(0, h - c)
        p.lineTo(c, h)
        
        # Top line
        p.lineTo(w - c, h)
        
        # Top-right chamfer
        p.lineTo(w, h - c)
        
        # Right line
        p.lineTo(w, c)
        
        # Bottom-right chamfer
        p.lineTo(w - c, 0)
        
        # Bottom line
        p.lineTo(c, 0)
        
        # Bottom-left chamfer
        p.lineTo(0, c)
        
        # Close path
        p.close()
        
        self.canv.drawPath(p, stroke=1, fill=0)
        
        # Add tech decorative accents (simple lines)
        self.canv.setStrokeColor(colors.darkorange)
        self.canv.setLineWidth(2)
        
        # Top decorative line
        self.canv.line(c + 10, h + 3, w - c - 10, h + 3)
        # Bottom decorative line
        self.canv.line(c + 10, -3, w - c - 10, -3)

        self.canv.restoreState()


def draw_racing_template(canvas: canvas.Canvas, doc: SimpleDocTemplate):
    """
    Draws the static background elements (Header, Footer, Checkers, Page Num).
    """
    canvas.saveState()
    w, h = A4
    
    # --- Colors ---
    racing_blue = colors.HexColor("#002244")  # Deep Navy Blue
    accent_orange = colors.orange
    
    # --- Header Bar ---
    header_h = 1.5 * cm
    canvas.setFillColor(racing_blue)
    canvas.rect(0, h - header_h, w, header_h, fill=1, stroke=0)
    
    # Header Accent Line (bottom of header)
    canvas.setStrokeColor(accent_orange)
    canvas.setLineWidth(2)
    canvas.line(0, h - header_h, w, h - header_h)

    # --- Footer Bar ---
    footer_h = 1.5 * cm
    canvas.setFillColor(racing_blue)
    canvas.rect(0, 0, w, footer_h, fill=1, stroke=0)
    
    # Footer Accent Line (top of footer)
    canvas.setStrokeColor(accent_orange)
    canvas.setLineWidth(2)
    canvas.line(0, footer_h, w, footer_h)

    # --- Checkered Flag Pattern ---
    # Draw simple white squares to simulate checkers on the bars
    check_size = 8
    
    # Top-Right Checkers
    rows = 3
    cols = 10
    start_x = w - (cols * check_size) - 10
    start_y = h - header_h + 5
    canvas.setFillColor(colors.white)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                canvas.rect(start_x + (c * check_size), start_y + (r * check_size), check_size, check_size, fill=1, stroke=0)

    # Bottom-Left Checkers (or alternating based on page num?)
    start_x = 10
    start_y = 5
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                canvas.rect(start_x + (c * check_size), start_y + (r * check_size), check_size, check_size, fill=1, stroke=0)

    # --- Page Number (Flame Icon Style) ---
    page_num = doc.page
    
    # Determine side based on page number (Even=Left, Odd=Right)
    # Note: doc.page starts at 1. 
    # Reference image: Page 4 (Left), Page 5 (Right).
    if page_num % 2 == 0:
        # Left (Bottom-Left)
        badge_x = 2 * cm
        badge_y = 0.75 * cm
    else:
        # Right (Bottom-Right)
        badge_x = w - 2 * cm
        badge_y = 0.75 * cm
        
    # Draw Flame/Wheel Badge Circle
    canvas.setFillColor(colors.white)
    canvas.setStrokeColor(colors.black)
    canvas.setLineWidth(1)
    canvas.circle(badge_x, badge_y, 12, fill=1, stroke=1)
    
    # Draw Number
    canvas.setFillColor(colors.black)
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawCentredString(badge_x, badge_y - 3.5, str(page_num))

    canvas.restoreState()


class PDFRenderer:
    def __init__(self) -> None:
        self.page_width, self.page_height = A4
        
        # Layout constants
        # Increased margins to account for the header/footer bars
        self.margin_top = 3.5 * cm
        self.margin_bottom = 3.5 * cm
        self.margin_side = 2.0 * cm
        
        # Reduced from 9.0cm to 8.0cm to prevent overflow/blank pages
        self.image_max_height = 8.0 * cm 
        
        # Styles
        self.styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle(
            "Title",
            parent=self.styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=28,
            textColor=colors.HexColor("#002244"), # Racing Blue
            spaceAfter=30,
            alignment=1,  # Center
        )
        self.page_text_style = ParagraphStyle(
            "PageText",
            parent=self.styles["Normal"],
            fontName="Helvetica",
            fontSize=16, # Larger for kids
            leading=21,  # Reduced from 24 to 21 to save vertical space
            spaceAfter=12,
            alignment=0,  # Left
            textColor=colors.black,
            leftIndent=1.0 * cm,  # Narrower text column
            rightIndent=1.0 * cm,
        )

    def render_book(self, book_spec: BookSpec, frames_dir: Path, output_path: Path) -> None:
        """Render the book specification to a PDF file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            title=book_spec.title,
            author="Vid2BedtimeStory",
            leftMargin=self.margin_side,
            rightMargin=self.margin_side,
            topMargin=self.margin_top,
            bottomMargin=self.margin_bottom,
        )

        story = []

        # Add title page
        story.extend(self._build_title_page(book_spec.title))

        # Add content pages
        total_pages = len(book_spec.pages)
        for i, page_spec in enumerate(book_spec.pages):
            story.extend(self._build_content_page(page_spec, frames_dir, is_last=(i == total_pages - 1)))

        # Apply the page template (onPage) to draw the static graphics
        doc.build(story, onFirstPage=draw_racing_template, onLaterPages=draw_racing_template)

    def _build_title_page(self, title: str) -> list:
        """Build the title page elements."""
        elements = []
        elements.append(Spacer(1, 4 * cm))
        elements.append(Paragraph(title, self.title_style))
        elements.append(PageBreak())
        return elements

    def _build_content_page(self, page_spec, frames_dir: Path, *, is_last: bool) -> list:
        """Build a content page with image and text."""
        elements = []

        # STRICT ALTERNATION:
        # Reference: Page 4 (Even/Left) -> Text Top, Image Bottom
        # Reference: Page 5 (Odd/Right) -> Image Top, Text Bottom
        
        is_even = (page_spec.page_index % 2 == 0)
        
        image_block = self._build_image_block(page_spec.page_index, frames_dir)
        text_block = self._build_text_block(page_spec.paragraphs)

        if is_even:
            # Text Top
            elements.extend(text_block)
            elements.append(Spacer(1, 0.5 * cm)) # Reduced from 1.0cm
            elements.extend(image_block)
        else:
            # Image Top
            elements.extend(image_block)
            elements.append(Spacer(1, 0.5 * cm)) # Reduced from 1.0cm
            elements.extend(text_block)

        # Wrap content in KeepTogether to prevent splitting
        # This treats the entire page content as an atomic unit.
        page_content = KeepTogether(elements)
        
        result = [page_content]
        if not is_last:
            result.append(PageBreak())

        return result

    def _build_image_block(self, page_index: int, frames_dir: Path) -> list:
        frame_path = frames_dir / f"page_{page_index:03d}.png"
        if not frame_path.exists():
            return [Paragraph("[Image not found]", self.page_text_style)]

        try:
            with Image.open(frame_path) as img:
                img_width, img_height = img.size

            # Calculate available width based on margins
            available_width = self.page_width - (2 * self.margin_side)
            
            # Constrain dimensions
            max_width = available_width
            max_height = self.image_max_height

            scale = min(max_width / img_width, max_height / img_height)
            display_width = img_width * scale
            display_height = img_height * scale
            
            # Use custom TechFrame Flowable instead of standard Image
            return [
                TechFrame(frame_path, display_width, display_height),
                Spacer(1, 0.2 * cm)
            ]
        except Exception as e:
            return [Paragraph(f"[Image loading error: {e}]", self.page_text_style)]

    def _build_text_block(self, paragraphs: list[str]) -> list:
        if not paragraphs:
            return [Paragraph("[No text content]", self.page_text_style)]
        
        elements = []
        for raw_p in paragraphs:
            if not raw_p.strip():
                continue
            markup_text = to_reportlab_markup(raw_p)
            elements.append(Paragraph(markup_text, self.page_text_style))
            # No explicit spacer needed here because self.page_text_style has 'spaceAfter=12'
            
        return elements


class PDFWriteError(Exception):
    """Raised when PDF cannot be written (e.g., file locked by Preview)."""
    pass


def render_pdf(book_spec: BookSpec, frames_dir: Path, output_path: Path) -> None:
    """
    Render a book spec to PDF with proper error handling.
    
    Raises:
        PDFWriteError: If the file cannot be written (e.g., locked by another app)
    """
    import os
    import tempfile
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file exists and record its modification time
    existing_mtime = None
    if output_path.exists():
        existing_mtime = output_path.stat().st_mtime
    
    # First, try writing to a temp file in the same directory to test permissions
    temp_path = output_path.with_suffix('.pdf.tmp')
    
    try:
        renderer = PDFRenderer()
        renderer.render_book(book_spec, frames_dir, temp_path)
        
        # Verify temp file was created
        if not temp_path.exists():
            raise PDFWriteError(f"PDF rendering failed: temp file was not created")
        
        # Now try to replace/create the target file
        # On macOS, if Preview has the file open, this rename may fail
        try:
            # Remove existing file first (required on Windows, good practice on macOS)
            if output_path.exists():
                output_path.unlink()
            temp_path.rename(output_path)
        except PermissionError as e:
            raise PDFWriteError(
                f"Cannot write to '{output_path}': file may be locked by another application "
                f"(e.g., Preview). Close the file and try again. Error: {e}"
            )
        except OSError as e:
            raise PDFWriteError(
                f"Cannot write to '{output_path}': {e}. "
                f"If Preview has the file open, close it and try again."
            )
        
        # Verify the file was actually updated
        if not output_path.exists():
            raise PDFWriteError(f"PDF file was not created at '{output_path}'")
        
        new_mtime = output_path.stat().st_mtime
        if existing_mtime is not None and new_mtime <= existing_mtime:
            raise PDFWriteError(
                f"PDF file at '{output_path}' was not updated (modification time unchanged). "
                f"The file may be locked by another application like Preview. "
                f"Close the file and try again."
            )
            
    except PDFWriteError:
        # Re-raise our custom errors
        raise
    except Exception as e:
        # Wrap other exceptions with helpful context
        raise PDFWriteError(f"PDF rendering failed: {e}") from e
    finally:
        # Clean up temp file if it still exists
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
