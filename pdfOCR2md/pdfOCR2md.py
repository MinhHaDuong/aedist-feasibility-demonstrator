"""pdfOCR2md.py

Minh Ha-Duong, CNRS, 2025-
License CC-BY-SA
"""

import sys
import os
import base64
from pdf2image import convert_from_path
from openai import OpenAI

MODEL = "gpt-4o"  # "gpt-4o-mini" is insufficient

SYSTEM_PROMPT = """You are an assistant that converts PDF page images to structured Markdown text.
Follow these rules:

Document Title and First Page:
- Use `#` for the document title only
- For Vietnamese administrative documents, combine all centered text into the title block:
  * The title block has a Type subblock and a Subject subblock
  * First line, in ALL CAPS, is the document type (BÁO CÁO, QUYẾT ĐỊNH, PHỤ LỤC, etc.)
  * After that, ALL centered lines is the document Subject subblock
  * The document Subject is NEVER empty
  * Title = DOCUMENT TYPE + <br> + document Subject
  * Title example 1: '#BÁO CÁO <br> Kế hoạch thực hiện Quy hoạch phát triển điện lực quốc gia thời kỳ 2021-2030, tầm nhìn đến năm 2050'
  * Title example 2 : '#QUYẾT ĐỊNH <br> Phê duyệt bổ sung, cập nhật Kế hoạch thực hiện Quy hoạch phát triển điện lực quốc gia thời kỳ 2021 - 2030, tầm nhìn đến năm 2050'
  * Title example 3: '#TỜ TRÌNH <br> Đề nghị ban hành bổ sung, cập nhật Kế hoạch thực hiện Quy hoạch phát triển điện lực quốc gia thời kỳ 2021-2030, tầm nhìn đến năm 2050'
- Format letterhead elements on the first page (institution, tagline, date, reference) as plain text with bold/italic
- Print letterhead elements before the title

Section Headers Throughout Document:
- Mark ALL section headers with `##`, `###` etc., even on the first page
- Section headers may appear anywhere, including first page after title
- Never skip marking a section header

Content Structure:
- Preserve document's logical hierarchy
- Keep all paragraphs, including those before section headers
- Convert lists to Markdown format with one blank line between items
- Fix OCR errors
- Include all text except page numbers
- Format footnotes as [^X] in text, [^X]: at bottom

Tables:
- Convert tables to properly formatted HTML with <table>
- Preserve table structure and formatting
- Handle complex tables organized in sections
"""


USER_PROMPT = """Here is the Markdown from the previous page (if empty, this is the first page):
{}

Now, convert the following base64-encoded page to Markdown, without adding explanations or comments.
Limit your response to the image content, without repeating the text above."""

def pdf_to_markdown_with_gptvision(pdf_path):
    """
    Converts PDF to images, encodes them in base64, and sends them
    to a vision model (e.g. gpt-4o) for Markdown conversion.
    
    To enhance consistency between pages, includes previously generated
    Markdown when converting the next page.
    
    The output is saved to a .md file with the same name as the PDF.
    """
    if not pdf_path.endswith(".pdf"):
        raise ValueError("File must be a PDF.")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File {pdf_path} not found.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OpenAI API key (OPENAI_API_KEY) not set in environment variables."
        )

    client = OpenAI(api_key=api_key)

    try:
        print("Converting PDF to images...")
        images = convert_from_path(pdf_path, dpi=300, fmt="jpeg")
    except Exception as e:
        raise RuntimeError(f"Error converting PDF to images: {str(e)}")

    # TODO: Check if there are many blank pages

    markdown_pieces = []
    previous_page_markdown = ""

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]

    for page_num, image in enumerate(images):
        print(f"Processing {base_name} page {page_num + 1}/{len(images)}...")

        temp_image_path = f"{base_name}_page_{page_num + 1}.jpeg"
        image.save(temp_image_path, "JPEG")

        with open(temp_image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
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
                            "url": f"data:image/jpeg;base64,{encoded_image}",
                        },
                    },
                ],
            },
        ]

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=3000,
            )
            print("Raw model response:", response)
            
            if response.choices:
                assistant_message = response.choices[0].message
                if assistant_message and hasattr(assistant_message, "content"):
                    markdown_text = assistant_message.content
                    markdown_text = markdown_text.replace("```markdown", "").replace("```", "")

                    markdown_pieces.append(markdown_text)
                    print(f"Page {page_num + 1}/{len(images)} converted successfully.")

                    previous_page_markdown = markdown_text
                else:
                    print(f"Unexpected model response for page {page_num + 1}: {assistant_message}")
            else:
                print("Unexpected model response: 'choices' is empty.")            

        except Exception as e:
            print(f"Error processing page {page_num + 1}: {str(e)}")
            continue
        finally:
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)

    # TODO: Handle multi-page tables
    # MAYBE:  include page numbers for RAG ?

    output_path = pdf_path.replace(".pdf", ".md")
    if os.path.exists(output_path):
        output_path = pdf_path.replace(".pdf", "_converted.md")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(markdown_pieces))

    return output_path

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pdfOCR2md.py file.pdf")
        sys.exit(1)

    try:
        output_file = pdf_to_markdown_with_gptvision(sys.argv[1])
        print(f"Conversion complete: {output_file}")
    except FileNotFoundError as fnfe:
        print(f"Error: {str(fnfe)}")
    except ValueError as ve:
        print(f"Validation error: {str(ve)}")
    except EnvironmentError as ee:
        print(f"Environment error: {str(ee)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
