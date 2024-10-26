import os
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from collections import Counter
import re


# Load the processor and model
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


def standardize_product_name(product):
    # Remove trailing punctuation
    product = product.strip(' .,;:')

    # Extract flavor from parentheses and append to product name
    match = re.search(r'\((.*)\)', product)
    if match:
        flavor = match.group(1)
        product = product.replace(f"({flavor})", "").strip() + " " + flavor

    return product


def refine_product_name(product):
    """
    Additional refinement to handle cases where the product name may not be cleanly extracted.
    For example, removing unnecessary tokens or fixing common issues.
    """
    # Removing common unnecessary terms or fixing common issues
    product = re.sub(r"\b(Extra|Extra Punch|Special)\b", "", product).strip()
    
    # You can add more product name refinements here if needed
    return product


def combine_similar_products(brand_product_counts):
    """
    Combines similar products by treating a product as the same if one name is a prefix of the other.

    Args:
        brand_product_counts (Counter): Counter object with brand-product tuples as keys.

    Returns:
        Counter: Updated counter with similar products combined.
    """
    combined_counts = Counter()

    # Iterate through the current counts and combine similar products
    for (brand, product), count in brand_product_counts.items():
        found_similar = False

        for (existing_brand, existing_product), existing_count in combined_counts.items():
            # Check if the current product is a prefix or variant of the existing one
            if brand == existing_brand and (
                product.startswith(existing_product) or existing_product.startswith(product)
            ):
                # Combine counts for similar products
                combined_counts[(existing_brand, existing_product)] += count
                found_similar = True
                break

        if not found_similar:
            combined_counts[(brand, product)] = count

    return combined_counts


def process_images_from_folder(folder_path):
    """
    Process images from a folder, extract brand and product names,
    standardize product names and count occurrences.

    Args:
        folder_path (str): Path to the folder containing images.

    Returns:
        Counter: A counter object with brand-product tuples as keys
        and occurrence counts as values.
    """

    brand_product_counts = Counter()

    # Iterate over all files in the folder
    for image_filename in os.listdir(folder_path):
        # Construct the full path to the image
        image_path = os.path.join(folder_path, image_filename)

        # Open the image
        image = Image.open(image_path)

        # Refined prompt with emphasis on flavor/variant detection
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

        # Apply the processor chat template to the image and text
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

        # Prepare the inputs for the model (image + prompt)
        inputs = processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)  # Ensure tensors are on the same device as the model

        # Generate output from the model
        output = model.generate(**inputs, max_new_tokens=50)

        # Decode the generated output to text
        generated_text = processor.decode(output[0], skip_special_tokens=True).strip()

        # Remove the prompt from the output
        prompt_length = len(processor.decode(inputs["input_ids"][0], skip_special_tokens=True).strip())
        output_text = generated_text[prompt_length:].strip()

        # Remove bold text markers
        output_text = output_text.replace("**", "")

        print(f"Output for {image_filename}: \"{output_text}\"")

        # Extract brand and product from output
        parts = output_text.split(", ")
        if len(parts) == 2:
            brand = parts[0].replace("Brand: ", "").strip()
            product = parts[1].replace("Product: ", "").strip()

            # Standardize product name and apply additional refinement
            product = standardize_product_name(product)
            product = refine_product_name(product)

            # Update count
            brand_product_counts[(brand, product)] += 1

            # Print updated count for the detected product
            print(f"Updated count for {brand}, {product}: {brand_product_counts[(brand, product)]}")

    # Combine similar products and return the final counts
    return combine_similar_products(brand_product_counts)

# Specify the folder path containing the images
folder_path = r"C:\Users\Administrator\drive D\Image_TextExtract\test\6"

# Process all images in the folder and get the brand/product count
brand_product_counts = process_images_from_folder(folder_path)

# Print brand/product counts
for (brand, product), count in brand_product_counts.most_common():
    print(f"{brand}, {product}: {count}")
