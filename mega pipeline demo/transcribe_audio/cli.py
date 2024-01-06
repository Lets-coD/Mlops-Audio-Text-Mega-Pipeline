"""
Module that contains the command line app.
"""
import os 
import io
import argparse
import shutil
from google.cloud import storage
from google.cloud import speech 
import ffmpeg
from tempfile import TemporaryDirectory
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, GPT2Config
from google.cloud import texttospeech
from googletrans import Translator

# Generate the inputs arguments parser
parser = argparse.ArgumentParser(description="Command description.")

gcp_project = "ac215-project"
bucket_name = "mega-pipeline-bucket"
input_audios = "input_audios"
text_prompts = "text_prompts"
text_paragraphs = "text_paragraphs"
text_translated = "text_translated"
output_audios = "output_audios"

translator = Translator()

def makedirs():
    os.makedirs(input_audios,exist_ok=True)
    os.makedirs(text_prompts,exist_ok=True)
    os.makedirs(text_paragraphs, exist_ok=True)
    os.makedirs(text_prompts, exist_ok=True)
    os.makedirs(output_audios, exist_ok=True)
    os.makedirs(text_translated, exist_ok=True)

def download(): 
    print("download")
    # Clear
    shutil.rmtree(input_audios, ignore_errors=True, onerror=None)
    makedirs()

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blobs =  bucket.list_blobs(prefix=input_audios + "/")
    for blob in blobs:
        print(blob.name)
        if not blob.name.endswith("/"):
            blob.download_to_filename(blob.name)

def downloadg():
    print("download")

    # Clear
    shutil.rmtree(text_prompts, ignore_errors=True, onerror=None)
    makedirs()

    storage_client = storage.Client(project=gcp_project)

    bucket = storage_client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=text_prompts + "/")
    for blob in blobs:
        print(blob.name)
        if blob.name.endswith(".txt"):
            blob.download_to_filename(blob.name)

def downloads():
    print("download")

    # Clear
    shutil.rmtree(text_translated, ignore_errors=True, onerror=None)
    makedirs()

    storage_client = storage.Client(project=gcp_project)

    bucket = storage_client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=text_translated + "/")
    for blob in blobs:
        print(blob.name)
        if blob.name.endswith(".txt"):
            blob.download_to_filename(blob.name)

def downloadtr():
    print("download")

    # Clear
    shutil.rmtree(text_paragraphs, ignore_errors=True, onerror=None)
    makedirs()

    storage_client = storage.Client(project=gcp_project)

    bucket = storage_client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=text_paragraphs + "/")
    for blob in blobs:
        print(blob.name)
        if blob.name.endswith(".txt"):
            blob.download_to_filename(blob.name)


def transcribe(): 
    print("transcribe")
    makedirs()
    # Speech client
    client = speech.SpeechClient()
    # Get the list of audio file audio_files = os.listdir (input_audios)
    audio_files = os.listdir(input_audios)
    for audio_path in audio_files:
        uuid= audio_path.replace(".mp3", "")
        audio_path = os.path.join(input_audios, audio_path) 
        text_file = os.path.join(text_prompts, uuid + ".txt")

        if os.path.exists(text_file):
            continue

        print("Transcribing:", audio_path)

        with TemporaryDirectory() as audio_dir:
            flac_path = os.path.join(audio_dir, "audio.flac") 
            stream = ffmpeg. input (audio_path)
            stream = ffmpeg.output(stream, flac_path) 
            ffmpeg.run(stream)


            with io.open(flac_path, "rb") as audio_file:
                content = audio_file.read()
                # Transcribe
            audio=speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(language_code="en-US") 
            operation = client.long_running_recognize(config=config, audio=audio) 
            response = operation.result(timeout=90) 
            print("response:", response) 
            text = "None"
            if len(response.results) > 0:
                text = response.results[0].alternatives[0].transcript 
                print(text)
            with open(text_file,"w") as f:
                # Save the transcription with open(text_File, "w") as f:
                f.write(text)

def generate():
    print("generate")
    makedirs()

    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # Model - Load pretrained GPT Language Model
    model = TFGPT2LMHeadModel.from_pretrained(
        "gpt2", pad_token_id=tokenizer.eos_token_id
    )

    # Get the list of text file
    text_files = os.listdir(text_prompts)

    for text_file in text_files:
        uuid = text_file.replace(".txt", "")
        file_path = os.path.join(text_prompts, text_file)
        paragraph_file = os.path.join(text_paragraphs, uuid + ".txt")

        if os.path.exists(paragraph_file):
            continue

        with open(file_path) as f:
            input_text = f.read()

        # Tokenize Input
        input_ids = tokenizer.encode(input_text, return_tensors="tf")
        print("input_ids", input_ids)

        # Generate output
        outputs = model.generate(input_ids, do_sample=True, max_length=100, top_k=50)
        paragraph = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Generate output
        outputs = model.generate(
            input_ids, max_length=50, num_beams=3, early_stopping=False
        )
        paragraph = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print("Generated text:")
        print(paragraph)

        # Save the transcription
        with open(paragraph_file, "w") as f:
            f.write(paragraph)


def synthesis():
    print("synthesis")
    makedirs()

    language_code = "hi-IN"

    # Get the list of text file
    text_files = os.listdir(text_translated)

    for text_file in text_files:
        filename = text_file.replace(".txt", "")
        file_path = os.path.join(text_translated, text_file)
        audio_file = os.path.join(output_audios, filename + ".mp3")

        if os.path.exists(audio_file):
            continue

        with open(file_path) as f:
            input_text = f.read()

        # Set the text input to be synthesized
        synthesis_input = texttospeech.SynthesisInput(text=input_text)

        # Build the voice request, select the language code ("en-US") and the ssml
        # voice gender ("neutral")
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,
        )

        # Select the type of audio file you want returned
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        # Perform the text-to-speech request on the text input with the selected
        # voice parameters and audio file type
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        # Save the audio file
        with open(audio_file, "wb") as out:
            # Write the response to the output file.
            out.write(response.audio_content)


def transcribe():
    print("transcribe")
    makedirs()

    # Speech client
    client = speech.SpeechClient()

    # Get the list of audio file
    audio_files = os.listdir(input_audios)

    for audio_path in audio_files:
        uuid = audio_path.replace(".mp3", "")
        audio_path = os.path.join(input_audios, audio_path)
        text_file = os.path.join(text_prompts, uuid + ".txt")

        if os.path.exists(text_file):
            continue

        print("Transcribing:", audio_path)
        with TemporaryDirectory() as audio_dir:
            flac_path = os.path.join(audio_dir, "audio.flac")
            stream = ffmpeg.input(audio_path)
            stream = ffmpeg.output(stream, flac_path)
            ffmpeg.run(stream)

            with io.open(flac_path, "rb") as audio_file:
                content = audio_file.read()

            # Transcribe
            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(language_code="en-US")
            operation = client.long_running_recognize(config=config, audio=audio)
            response = operation.result(timeout=90)
            print("response:", response)
            text = "None"
            if len(response.results) > 0:
                text = response.results[0].alternatives[0].transcript
                print(text)

            # Save the transcription
            with open(text_file, "w") as f:
                f.write(text)


def translate():
    print("translate")
    makedirs()

    # Get the list of text file
    text_files = os.listdir(text_paragraphs)

    for text_file in text_files:
        uuid = text_file.replace(".txt", "")
        file_path = os.path.join(text_paragraphs, text_file)
        translated_file = os.path.join(text_translated, uuid + ".txt")

        if os.path.exists(translated_file):
            continue

        with open(file_path) as f:
            input_text = f.read()

        results = translator.translate(input_text, src="en", dest="hi")

        print(results.text)

        # Save the translation
        with open(translated_file, "w") as f:
            f.write(results.text)


def upload(): 
    print("upload") 
    makedirs()
    # Upload to bucket
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    # Get the list of text file 
    text_files = os.listdir(text_prompts)
    for text_file in text_files:
        file_path = os.path.join(text_prompts, text_file)
        destination_blob_name = file_path 
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(file_path)


def uploadtr():
    print("upload")
    makedirs()

    # Upload to bucket
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Get the list of text file
    text_files = os.listdir(text_translated)

    for text_file in text_files:
        file_path = os.path.join(text_translated, text_file)

        destination_blob_name = file_path
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(file_path)

def uploads():
    print("upload")
    makedirs()

    # Upload to bucket
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Get the list of files
    audio_files = os.listdir(output_audios)

    for audio_file in audio_files:
        file_path = os.path.join(output_audios, audio_file)

        destination_blob_name = file_path
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(file_path)

def uploadg():
    print("upload")
    makedirs()

    # Upload to bucket
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Get the list of text file
    text_files = os.listdir(text_paragraphs)

    for text_file in text_files:
        file_path = os.path.join(text_paragraphs, text_file)

        destination_blob_name = file_path
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(file_path)

def main(args=None):

    print("Args:", args)

    if args.download:
        download()
    if args.downloadg:
        downloadg()
    if args.downloads:
        downloads()
    if args.downloadtr:
        downloadtr()        
    if args.transcribe:
        transcribe()
    if args.upload:
        upload()
    if args.uploadtr:
        uploadtr()
    if args.uploads:
        uploads()
    if args.uploadg:
        uploadg()
    if args.translate:
        translate()
    if args.synthesis:
        synthesis()
    if args.generate:
        generate()

if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal 'python cli.py --help', it will provide the description
    parser = argparse.ArgumentParser(description='Transcribe audio file to text')

    parser.add_argument("-d", "--download", action='store_true',
                        help="Download audio files from GCS bucket")

    parser.add_argument("-dg", "--downloadg", action='store_true',
                        help="Download audio files from GCS bucket")

    parser.add_argument("-ds", "--downloads", action='store_true',
                        help="Download audio files from GCS bucket")

    parser.add_argument("-dtr", "--downloadtr", action='store_true',
                        help="Download audio files from GCS bucket")



    parser.add_argument(
        "-g", "--generate", action="store_true", help="Generate a text paragraph"
    )
    parser.add_argument(
        "-s", "--synthesis", action="store_true", help="Synthesis audio"
    )
    parser.add_argument("-tr", "--translate", action="store_true", help="Translate text")
    parser.add_argument("-t", "--transcribe", action='store_true',
                        help="Transcribe audio files to text")



    parser.add_argument("-u", "--upload", action='store_true',
                        help="Upload transcribed text to GCS bucket")
    parser.add_argument("-utr", "--uploadtr", action='store_true',
                        help="Upload transcribed text to GCS bucket")
    parser.add_argument("-us", "--uploads", action='store_true',
                        help="Upload transcribed text to GCS bucket")
    parser.add_argument("-ug", "--uploadg", action='store_true',
                        help="Upload transcribed text to GCS bucket")
    args = parser.parse_args()

    main(args)
