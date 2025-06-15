import json
import os

import imageio
from moviepy.editor import VideoFileClip
from openai import OpenAI


def extract_audio(video_path, audio_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path)


def extract_text(audio_path, text_path):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1",
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"]
        )
        with open(text_path, "w") as text_file:
            text_file.write(transcription.text)

        with open(text_path.replace(".txt", ".json"), "w") as json_file:
            json_file.write(transcription.json())


def build_instructions(text_path, instructions_path):
    with open(text_path, "r") as text_file:
        text = text_file.read()

    system_prompt = """
You are a helpful assistent in charge to create a manual of procedures using the rules present in the user description.

Your answer should be returned as a sequence of steps to accomplish the described task and the keyword related to the task present inside the description.

The output should be in the following format:

- Code: the product code
- Steps:
    1. A sequence of steps
- Keywords
    1. keyword

For example, given the following user content:

---
Essa é a descrição de um sofá de código 123. Para avaliar um sofá, você deve verificar se o tecido está totalmente fixado e se os braços e encosto estão fixados corretamente.
---

The answer should be:

- Code: 123
- Steps:
    1. Verificar se o tecido está fixado;
    2. Verifique se os braços estão fixados;
    3. Verificar se o encosto está fixado.
- Keywords:
    1. tecido;
    2. braços;
    3. encosto.

Return the answer in the portuguese language.
"""
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": text,
        }
    ]
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    with open(instructions_path, "w") as instructions_file:
        instructions_file.write(response.choices[0].message.content)

    with open(instructions_path.replace(".txt", ".json"), "w") as json_file:
        json_file.write(response.json())


def crop_images(
    instructions_path,
    text_json_path,
    video_path,
    images_folder,
):
    with open(text_json_path, "r") as text_json_file:
        text_json = json.load(text_json_file)

    with open(instructions_path, "r") as instructions_file:
        instructions = instructions_file.read()
        instructions_rules = instructions.split("- Keywords:")[1].strip()

    keywords = []
    for r in instructions_rules.split("\n"):
        keyword = r.split(" ")[-1].replace(";", "").replace(".", "")
        keywords.append(keyword)

    print(keywords)

    keywords_found = []
    markers = []
    start = 0
    for k in keywords:
        for w in text_json["words"][start:]:
            start += 1
            if w["word"]  == k:
                keywords_found.append(k)
                markers.append(w["end"])
                break

    print(markers)

    clip = VideoFileClip(video_path)
    for i, m in enumerate(markers):
        frame = clip.get_frame(m)
        imageio.imwrite(f"{images_folder}/{i + 1}.png", frame)


if __name__ == "__main__":
    # extract_audio(
    #     "contrib/data/caneta.webm",
    #     "contrib/data/caneta.mp3",
    # )
    # extract_text(
    #     "contrib/data/caneta.mp3",
    #     "contrib/data/caneta.txt",
    # )
    # build_instructions(
    #     "contrib/data/caneta.txt",
    #     "contrib/data/caneta_instructions.txt",
    # )
    crop_images(
        "contrib/data/caneta_instructions.txt",
        "contrib/data/caneta.json",
        "contrib/data/caneta.webm",
        "contrib/data/images",
    )
