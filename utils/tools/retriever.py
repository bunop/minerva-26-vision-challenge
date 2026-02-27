import ast
import json
import pandas as pd


class Retriever:
    def __init__(self, top_k=1):
        self.top_k = top_k

        print("Loading KB...")
        wiki_KB_path = "./data/evqa/encyclopedic_kb_wiki.json"
        with open(wiki_KB_path, "r") as f:
            self.wikipedia = json.load(f)
        print("KB loaded.")

        self.google_lens_path = "./data/evqa/lens_entities.csv"
        self.google_lens_data = pd.read_csv(self.google_lens_path)

    def retrieve(self, dataset_image_id):
        # implement retrieval of wiki urls based on dataset_image_id from google lens data
        # this is a list of objects
        wiki_urls = self.google_lens_data[
            self.google_lens_data["dataset_image_id"] == dataset_image_id
        ]["lens_wiki_urls"].values[0]
        wiki_urls = ast.literal_eval(wiki_urls)

        # print(f"Retrieved wiki urls for image {dataset_image_id}: {wiki_urls}")

        # extract text from the retrieved wiki urls and concatenate them to form the context
        # I can have multiple wiki urls, so I will concatenate the text from the top k wiki urls
        context = ""
        for i in range(self.top_k):
            if i > len(wiki_urls):
                break

            url = wiki_urls[i]
            # print(f"Retrieving context from wiki url: {url}")
            section_texts = self.wikipedia.get(url, {"section_texts": []}).get(
                "section_texts", []
            )
            context += " ".join(section_texts)

        return context
