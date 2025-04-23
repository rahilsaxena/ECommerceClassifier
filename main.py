import argparse
from src.inference import classify_text

def main():
    parser = argparse.ArgumentParser(description="Classify E-commerce Inquiry")
    parser.add_argument('--text', type=str, required=True, help='Text to classify')
    args = parser.parse_args()
    category = classify_text(args.text)
    print(f"Predicted Category: {category}")

if __name__ == "__main__":
    main()
