"""pdfOCR2md.py

Minh Ha-Duong, CNRS, 2025-
License CC-BY-SA
"""

import sys
import os
import base64
import re
import datetime
import platform

from pdf2image import convert_from_path
from openai import OpenAI

MODEL = "gpt-4o"
# MODEL = "gpt-4o-mini"  makes too many OCR errors
MAX_TOKENS = 4096  # Default output token limit


SYSTEM_PROMPT = """You are an assistant that converts PDF page images to structured Markdown text.
Follow these rules:

Text Formatting:
- Fix any OCR errors
- Include all text except page numbers

Document Title and First Page:
- Use `#` for the document title only
- For Vietnamese administrative documents:
  * Print order of letterhead elements: country, tagline, ministry, reference number, place and date
  * Letterhead elements BEFORE the title.
  * The title block has a Type subblock and a Subject subblock
  * First line, in ALL CAPS, is the document type (BÁO CÁO, QUYẾT ĐỊNH, PHỤ LỤC, etc.)
  * After that, ALL centered lines is the document Subject subblock
  * The document Subject is NEVER empty
  * Title = DOCUMENT TYPE + <br> + document Subject
  * Title example 1: '#BÁO CÁO <br> Kế hoạch thực hiện Quy hoạch phát triển điện lực quốc gia thời kỳ 2021-2030, tầm nhìn đến năm 2050'
  * Title example 2 : '#QUYẾT ĐỊNH <br> Phê duyệt bổ sung, cập nhật Kế hoạch thực hiện Quy hoạch phát triển điện lực quốc gia thời kỳ 2021 - 2030, tầm nhìn đến năm 2050'
  * Title example 3: '#TỜ TRÌNH <br> Đề nghị ban hành bổ sung, cập nhật Kế hoạch thực hiện Quy hoạch phát triển điện lực quốc gia thời kỳ 2021-2030, tầm nhìn đến năm 2050'

Section Headers Throughout Document:
- Mark ALL section headers with `##`, `###` etc., even on the first page
- Section headers may appear anywhere, including first page after title
- Never skip marking a section header

Content Structure:
- Preserve document's logical hierarchy
- Keep all paragraphs, including those before section headers
- Use proper Markdown list formatting with one blank line between items
- Format footnotes as [^X] in text with corresponding [^X]: at bottom

Tables:
- Always use HTML table syntax (<table>, <tr>, <td>), never Markdown tables
- Preserve all table formatting, including merged cells and alignments
- For tables that appear to continue from previous page (no caption, partial content):
  * Insert <!-- TableContinues --> before the opening <table> tag
  * Maintain same column structure and alignment
  * Include table headers if present on the page
  * Omit <caption> tag since it's a continuation
- Keep all table section headings and subheadings within the table structure
"""


USER_PROMPT = """Here is the Markdown from the previous page (if empty, this is the first page):
{}

Now, convert the following base64-encoded page to Markdown, without adding explanations or comments.
Limit your response to the image content, without repeating the text above."""


def validate_inputs(pdf_path):
    """Validate input file and environment."""
    if not pdf_path.endswith(".pdf"):
        raise ValueError("File must be a PDF.")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File {pdf_path} not found.")
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OpenAI API key not set in environment variables.")


def get_output_path(pdf_path):
    """Determine output markdown file path."""
    output_path = pdf_path.replace(".pdf", ".md")
    if os.path.exists(output_path):
        output_path = pdf_path.replace(".pdf", "_converted.md")
    return output_path


def process_model_response(response, page_num, total_pages):
    """Extract and clean markdown text from model response."""
    if not response.choices:
        raise ValueError("Unexpected model response: 'choices' is empty.")

    assistant_message = response.choices[0].message
    if not (assistant_message and hasattr(assistant_message, "content")):
        raise ValueError(f"Unexpected model response for page {page_num + 1}")

    markdown_text = assistant_message.content

    # Remove code block markers
    markdown_text = re.sub(r"^```(?:markdown|html)?[ \t]*\n", "", markdown_text)
    markdown_text = re.sub(r"\n```[ \t]*$", "", markdown_text)

    # Standardize table indentation to 2 spaces
    markdown_text = re.sub(r"^(\s*)<tr", r"  <tr", markdown_text, flags=re.MULTILINE)
    markdown_text = re.sub(r"^(\s*)<td", r"    <td", markdown_text, flags=re.MULTILINE)
    markdown_text = re.sub(r"^(\s*)<th", r"    <th", markdown_text, flags=re.MULTILINE)

    # Strip trailing whitespaces
    markdown_text = "\n".join(line.rstrip() for line in markdown_text.splitlines())

    return f"<!-- PDF page {page_num + 1} -->\n" + markdown_text


def metadata_comment():
    """Create conversion metadata comment."""
    return f"""

<!-- Converted from PDF using:
Command: python {' '.join(sys.argv)}
Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Source: {os.path.basename(sys.argv[1])}
Platform: {platform.platform()}
Python: {platform.python_version()}
Model: {MODEL}
-->"""


def pdf_to_markdown(pdf_path):
    """Convert PDF to Markdown using GPT Vision API."""
    validate_inputs(pdf_path)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]

    print("Converting PDF to images...")
    images = convert_from_path(pdf_path, dpi=300, fmt="jpeg")

    markdown_pieces = []
    previous_page_markdown = ""

    for page_num, image in enumerate(images):
        print(f"Processing {base_name} page {page_num + 1}/{len(images)}...")

        # Save temporary image
        temp_image_path = f"{base_name}_page_{page_num + 1}.jpeg"
        image.save(temp_image_path, "JPEG")

        try:
            # Process page
            with open(temp_image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": USER_PROMPT.format(previous_page_markdown),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            },
                        },
                    ],
                },
            ]

            response = client.chat.completions.create(
                model=MODEL, messages=messages, max_tokens=MAX_TOKENS
            )

            markdown_text = process_model_response(response, page_num, len(images))
            markdown_pieces.append(markdown_text)
            previous_page_markdown = markdown_text

            print(f"Page {page_num + 1}/{len(images)} converted successfully.")

        except Exception as e:
            print(f"Error processing page {page_num + 1}: {str(e)}")
            continue
        finally:
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)

    return "\n".join(markdown_pieces)


# TODO: Deal with running headers
# TODO: Merge continued tables. Could be as simple as stripping </table>\n<!-- PDF Page ## -->\n<!-- PageContinues --><table>. Use a string constant for the PageContinues marker.

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pdfOCR2md.py file.pdf")
        sys.exit(1)

    try:
        pdf_path = sys.argv[1]
        result = pdf_to_markdown(pdf_path)
        output_path = get_output_path(pdf_path)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result)
            f.write(metadata_comment())
        print(f"Conversion complete: {output_path}")
    except FileNotFoundError as fnfe:
        print(f"Error: {str(fnfe)}")
    except ValueError as ve:
        print(f"Validation error: {str(ve)}")
    except EnvironmentError as ee:
        print(f"Environment error: {str(ee)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
