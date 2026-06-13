import os
import numpy as np
import cv2
from dotenv import load_dotenv
from supabase import create_client

class digimon_db:
    def __init__(self):
        load_dotenv()
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_KEY")
        self.bucket = os.getenv("BUCKET_NAME")
        self.profile_dir = os.getenv("TIMESTRANGER_PROFILE_DIR")
        self.schema = os.getenv("TIMESTRANGER_SCHEMA")

    def connect(self):
        self.supabase = create_client(self.url, self.key)
        self.supabase.auth.sign_in_with_password({
            "email": os.getenv("SUPABASE_ID"),
            "password": os.getenv("SUPABASE_PASSWORD")
        })
    
    def disconnect(self):
        self.supabase.auth.sign_out()

    def load_generations(self, locale):
        self.generations = (
            self.supabase.schema(f"{self.schema}")
            .table(f"generations")
            .select("id, generation_translations(name)")
            .eq(f"generation_translations.language_code", locale)
            .execute()
            ).data
        return [
            {
                "id": generation["id"],
                "name": generation["generation_translations"][0]["name"]
            }
            for generation in self.generations
        ]
    def load_attributes(self, locale):
        self.attributes = (
            self.supabase.schema(f"{self.schema}")
            .table(f"attributes")
            .select("id, attribute_translations(name)")
            .eq(f"attribute_translations.language_code", locale)
            .execute()
            ).data
        return [
            {
                "id": attribute["id"],
                "name": attribute["attribute_translations"][0]["name"]
            }
            for attribute in self.attributes
        ]
    
    def update_digimon_profile_image(self, dex_number, image_url):
        return self.supabase.schema(f"{self.schema}").table("digimons").update({
            "image_url": image_url
        }).match({"dex_number": dex_number}).execute()
    
    def update_digimon_profile_attribute(self, dex_number, attribute_id):
        return self.supabase.schema(f"{self.schema}").table("digimons").update({
            "attribute_id": attribute_id
        }).match({"dex_number": dex_number}).execute()
    
    def update_digimon_profile_generation(self, dex_number, generation_id):
        return self.supabase.schema(f"{self.schema}").table("digimons").update({
            "generation_id": generation_id
        }).match({"dex_number": dex_number}).execute()
    
    def insert_digimon_name_translation(self, dex_number, language_code, name):
        return self.supabase.schema(f"{self.schema}").table("digimon_translations").insert({
            "digimon_dex_number": dex_number,
            "language_code": language_code,
            "name": name
        }).execute()
    
    def upload_digimon_profile_image(self, image_path, target_path):
        with open(image_path, "rb") as f:
            file_data = f.read()

        return self.supabase.storage.from_(self.bucket).upload(
            f"{os.getenv('TIMESTRANGER_PROFILE_DIR')}/{target_path}", 
            file_data, 
            file_options = {"content-type":self.get_mime_type(image_path)}
            ).path
    
    def get_mime_type(self, file_name):
        ext = os.path.splitext(file_name)[1].lower()
        mime_types = {
            '.webp': 'image/webp',
            '.png': 'image/png',
        }

        return mime_types.get(ext, 'image/jpeg')
