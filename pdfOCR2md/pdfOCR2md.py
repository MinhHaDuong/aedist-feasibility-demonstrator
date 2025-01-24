import sys
import os
import base64
from pdf2image import convert_from_path
from openai import OpenAI


def pdf_to_markdown_with_gptvision(pdf_path):
    """
    Convertit un PDF en images, les encode en base64, et les envoie
    à un modèle vision (ex. gpt-4o-mini) pour obtenir une conversion en Markdown.

    Le contenu retourné est ensuite sauvegardé dans un fichier .md
    portant le même nom que le PDF.
    """

    # Vérification de l'extension du fichier
    if not pdf_path.endswith(".pdf"):
        raise ValueError("Le fichier doit être un PDF.")

    # Vérification si le fichier existe
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Le fichier {pdf_path} est introuvable.")

    # Vérification de la clé API OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "La clé API OpenAI (OPENAI_API_KEY) n'est pas définie dans les variables d'environnement."
        )

    # Initialisation de l'API OpenAI
    client = OpenAI(api_key=api_key)

    # Convertir le PDF en images
    try:
        print("Conversion du PDF en images...")
        images = convert_from_path(pdf_path, dpi=300, fmt="jpeg")
    except Exception as e:
        raise RuntimeError(
            f"Erreur lors de la conversion du PDF en images : {str(e)}"
        )

    markdown_pieces = []

    # Traiter chaque image (page) et interroger le modèle Vision pour la conversion en Markdown
    for page_num, image in enumerate(images):
        print(f"Traitement de la page {page_num + 1}/{len(images)}...")

        # Sauvegarde temporaire de l'image
        temp_image_path = f"page_{page_num + 1}.jpeg"
        image.save(temp_image_path, "JPEG")

        # Encoder l'image en base64
        with open(temp_image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        # Préparer le message à envoyer au modèle
        # On suit la forme décrite dans "doc on vision.txt" :
        #   {
        #       "role": "user",
        #       "content": [
        #           {"type": "text", "text": "Texte à envoyer"},
        #           {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64, ..."}}
        #       ]
        #   }
        messages = [
            {
                "role": "system",
                "content": (
                    "Tu es un assistant qui convertit des images de pages PDF en texte Markdown structuré. "
                    "Suis ces règles :\n"
                    "- Identifie les titres et sous-titres, et utilise les `#`, `##`, etc., pour les hiérarchiser.\n"
                    "- Transforme les listes (numérotées ou à puces) en listes Markdown.\n"
                    "- Si le texte contient des tableaux, formate-les en tableaux Markdown.\n"
                    "- Les pages seront traitées dans l'ordre. Conserve la structure hiérarchique et la logique du document.\n"
                    "- Corrige les erreurs évidentes dues à l'OCR.\n"
                    "- Enlève la numérotation des pages."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Convertis en Markdown, sans ajouter d'explications ou de commentaires, "
                            "le contenu de la page suivante, encodée en base64."
       
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}",
                            # Optionnel : vous pouvez spécifier "detail": "low" ou "detail": "high" selon vos besoins
                            # "detail": "low"
                            # "detail": "high"
                        },
                    },
                ],
            },
        ]

        try:
            # Appel au modèle (exemple : gpt-4o-mini selon la doc vision)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=2000,  # Ajustez selon vos besoins
            )
            print("Réponse brute du modèle :", response)
            
            if response.choices:
                # Dans les versions récentes du SDK, response.choices[0].message est un objet ChatCompletionMessage
                assistant_message = response.choices[0].message
                if assistant_message and hasattr(assistant_message, "content"):
                    markdown_text = assistant_message.content
                    
                    # -- Nettoyage : retrait des balises code fences --
                    markdown_text = markdown_text.replace("```markdown", "")
                    markdown_text = markdown_text.replace("```", "")

                    markdown_pieces.append(markdown_text)
                    print(f"Page {page_num + 1}/{len(images)} convertie avec succès.")
                else:
                    print(
                        f"Réponse inattendue du modèle pour la page {page_num + 1} : {assistant_message}"
                    )
            else:
                print("Réponse inattendue du modèle : 'choices' est vide.")            

        except Exception as e:
            print(f"Erreur lors du traitement de la page {page_num + 1} : {str(e)}")
            continue
        finally:
            # Supprimer l'image temporaire
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)

    # Sauvegarde du résultat
    output_path = pdf_path.replace(".pdf", ".md")
    if os.path.exists(output_path):
        output_path = pdf_path.replace(".pdf", "_converted.md")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(markdown_pieces))

    return output_path


if __name__ == "__main__":
    # Vérification des arguments de la ligne de commande
    if len(sys.argv) != 2:
        print("Usage: python pdf2md.py fichier.pdf")
        sys.exit(1)

    try:
        # Conversion du PDF en Markdown en utilisant le modèle Vision
        output_file = pdf_to_markdown_with_gptvision(sys.argv[1])
        print(f"Conversion terminée : {output_file}")
    except FileNotFoundError as fnfe:
        print(f"Erreur : {str(fnfe)}")
    except ValueError as ve:
        print(f"Erreur de validation : {str(ve)}")
    except EnvironmentError as ee:
        print(f"Erreur d'environnement : {str(ee)}")
    except Exception as e:
        print(f"Une erreur inattendue s'est produite : {str(e)}")
