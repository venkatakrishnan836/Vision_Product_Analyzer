import os
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from collections import Counter
import re

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


def standardize_product_name(product):
    product = product.strip(' .,;:')

    match = re.search(r'\((.*)\)', product)
    if match:
        flavor = match.group(1)
        product = product.replace(f"({flavor})", "").strip() + " " + flavor

    return product


def refine_product_name(product):
    product = re.sub(r"\b(Extra|Extra Punch|Special)\b", "", product).strip()
    
    return product


def combine_similar_products(brand_product_counts):

    combined_counts = Counter()

    for (brand, product), count in brand_product_counts.items():
        found_similar = False

        for (existing_brand, existing_product), existing_count in combined_counts.items():
            if brand == existing_brand and (
                product.startswith(existing_product) or existing_product.startswith(product)
            ):
                combined_counts[(existing_brand, existing_product)] += count
                found_similar = True
                break

        if not found_similar:
            combined_counts[(brand, product)] = count

    return combined_counts


def process_images_from_folder(folder_path):
    brand_product_counts = Counter()

    for image_filename in os.listdir(folder_path):
        
        image_path = os.path.join(folder_path, image_filename)

        image = Image.open(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": """
                        DETECT BRAND NAME AND PRODUCT NAME, INCLUDING ANY FLAVOR OR VARIANT, FROM THE IMAGE.
                        
                        PROVIDE OUTPUT IN THE FORMAT:
                        "Brand: [BRAND NAME], Product: [PRODUCT NAME (including flavor or variant)]"

                        ONLY PROVIDE THE PRODUCT NAME WITHOUT ANY EXPLANATION OR ADDITIONAL TEXT.
                        """
                    }
                ]
            }
        ]

        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)

        output = model.generate(**inputs, max_new_tokens=50)

        generated_text = processor.decode(output[0], skip_special_tokens=True).strip()

        prompt_length = len(processor.decode(inputs["input_ids"][0], skip_special_tokens=True).strip())
        output_text = generated_text[prompt_length:].strip()

        output_text = output_text.replace("**", "")

        print(f"Output for {image_filename}: \"{output_text}\"")

        parts = output_text.split(", ")
        if len(parts) == 2:
            brand = parts[0].replace("Brand: ", "").strip()
            product = parts[1].replace("Product: ", "").strip()

            product = standardize_product_name(product)
            product = refine_product_name(product)

            brand_product_counts[(brand, product)] += 1

            print(f"Updated count for {brand}, {product}: {brand_product_counts[(brand, product)]}")

    return combine_similar_products(brand_product_counts)

# Specify the folder path containing the images
folder_path = "Folder Path"

brand_product_counts = process_images_from_folder(folder_path)

for (brand, product), count in brand_product_counts.most_common():
    print(f"{brand}, {product}: {count}")
