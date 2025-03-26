import gradio as gr
import pyaudio
import wave
import threading
import whisper
import noisereduce as nr
import numpy as np
from transformers import pipeline
import tempfile
import torch
import os
from langdetect import detect
from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.pagesizes import letter
from textwrap import wrap
import soundfile as sf
import re

# missing stop recording button
# Audio Recording Variables
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
recording = False
frames = []
transcription_result = ""
summary_result = ""


def start_recording():

    global recording, frames
    recording = True
    frames = []

    def record():
        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        while recording:
            data = stream.read(CHUNK)
            frames.append(data)
        stream.stop_stream()
        stream.close()
        audio.terminate()
        process_audio()

    threading.Thread(target=record, daemon=True).start()
    return "Recording in progress...", gr.update(visible=True), gr.update(visible=False), gr.update(variant="primary")


def stop_recording():
    global recording
    recording = False
    return "Processing... Please wait.", gr.update(visible=False), gr.update(visible=True), gr.update(
        variant="secondary")


def process_audio():
    global transcription_result, summary_result, frames

    transcription_result = ""
    summary_result = ""

    # Force CPU if CUDA error is encountered
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    audio = pyaudio.PyAudio()

    with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as tmpfile:
        audio_data = np.frombuffer(b"".join(frames), dtype=np.int16)
        reduced_noise = nr.reduce_noise(y=audio_data, sr=RATE)
        sf.write(tmpfile.name, reduced_noise, RATE, format='FLAC')

    try:
        model = whisper.load_model("small").to(device)
        transcription_result = transcribe_audio(tmpfile.name, model)
        summary_result = summarize_text(transcription_result)

    except RuntimeError as e:
        if "CUDA error: device-side assert triggered" in str(e):
            print("CUDA error detected. Switching to CPU...")

            # Reset CUDA memory before switching
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

            # Force CPU usage
            device = "cpu"
            model = whisper.load_model("small").to(device)

            transcription_result = transcribe_audio(tmpfile.name, model)
            summary_result = summarize_text(transcription_result)

    return transcription_result, summary_result


def transcribe_audio(audio_file, model):
    result = model.transcribe(audio_file, language=None)
    transcription_with_timestamps = "\n".join([
        f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}"
        for segment in result["segments"]])

    return transcription_with_timestamps


def remove_duplicates(summary_text):
    lines = summary_text.split("\n")
    return "\n".join(list(dict.fromkeys(lines)))


def summarize_text(transcript):
    # Remove timestamps
    cleaned_transcript = re.sub(r"\[\d+\.\d+s - \d+\.\d+s\]", "", transcript).strip()

    # Split transcript into sentences
    sentences = cleaned_transcript.split(". ")

    # Select only 1/3rd of the text for summarization
    reduced_text = ". ".join(sentences[:len(sentences) // 3])

    if len(reduced_text) < 200:  # If too short, take half instead
        reduced_text = ". ".join(sentences[:len(sentences)])

    # Load summarizer
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn",
                          device=0 if torch.cuda.is_available() else -1)

    # Summarize the reduced text
    summary = summarizer(reduced_text, max_length=250, min_length=100, do_sample=False, num_beams=8)[0]["summary_text"]

    # Remove duplicate sentences (if any)
    summary = remove_duplicates(summary)

    # Convert to bullet points
    bullet_points = "\n".join([f"- {point.strip()}" for point in summary.split(". ") if len(point) > 5])

    return bullet_points


def generate_pdf():
    global transcription_result, summary_result

    pdf_filename = "meeting_notes.pdf"
    c = canvas.Canvas(pdf_filename, pagesize=letter)

    # Register Fonts
    hindi_font_path = os.path.join(BASE_DIR, "fonts", "NotoSansDevanagari_Condensed-Black.ttf")
    english_font_path = os.path.join(BASE_DIR, "fonts", "DejaVuSans.ttf")

    pdfmetrics.registerFont(TTFont("HindiFont", hindi_font_path))
    pdfmetrics.registerFont(TTFont("EnglishFont", english_font_path))

    # Set Page Layout
    page_width, page_height = letter
    margin = 50
    line_height = 14
    text_width = page_width - (2 * margin)
    start_y = page_height - 80  # Start position for text

    def set_font(text, size=12):
        """Set appropriate font based on detected language."""
        try:
            lang = detect(text)
            c.setFont("HindiFont", size) if lang == "hi" else c.setFont("EnglishFont", size)
        except:
            c.setFont("EnglishFont", size)

    def add_page(title):
        """Adds a new page and resets text position."""
        c.showPage()  # Create new page
        set_font("Meeting Notes", size=22)
        c.drawString(200, page_height - 50, "Meeting Notes")
        set_font(title, size=14)
        c.drawString(margin, page_height - 80, title)
        return page_height - 100  # Reset y-position

    def draw_wrapped_text(text, x, y, max_width=text_width, line_height=14, section_title=""):
        """Handles text wrapping & pagination correctly."""
        lines = []
        for para in text.split("\n"):
            wrapped_lines = wrap(para, width=80)
            lines.extend(wrapped_lines)

        for line in lines:
            if y < margin:  # If text reaches the bottom, move to next page
                y = add_page(section_title)
            c.drawString(x, y, line)
            y -= line_height  # Move to next line
        return y

    # Add Title & Meeting Info
    set_font("Meeting Notes", size=22)
    c.drawString(200, start_y, "Meeting Notes")
    start_y -= 30

    set_font("Meeting Title : [Add Title] ", size=12)
    c.drawString(margin, start_y, "Meeting Title): [Add Title]")
    c.drawString(margin, start_y - 20, "बैठक स्थान (Meeting Location): [स्थान जोड़ें]")
    c.drawString(margin, start_y - 40, "उपस्थित लोग (Attendees): [उपस्थित लोगों को जोड़ें]")
    start_y -= 70

    # **Fix for Transcription Overflow**
    set_font("Transcription:", size=14)
    c.drawString(margin, start_y, "Transcription:")
    set_font(transcription_result, size=12)
    start_y -= 20
    start_y = draw_wrapped_text(transcription_result, margin, start_y, section_title="Transcription:")

    # **Fix for Summary Overflow**
    set_font("Summary:", size=14)
    c.drawString(margin, start_y - 20, "Summary:")
    set_font(summary_result, size=12)
    start_y = draw_wrapped_text(summary_result, margin, start_y - 40, section_title="Summary:")

    # Save PDF
    c.save()
    return pdf_filename


def download_pdf():
    global transcription_result, summary_result
    pdf_path = generate_pdf()
    return pdf_path


with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown(
        """
        # 🎙️ AI-Meeting Minutes
        ### Automatically Transcribe and Summarize your meetings(Both Sides) with AI 
         Works with Zoom , Meet , Skype , Teams , Whatsapp call...
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Recording Controls")
            with gr.Group():
                with gr.Row():
                    start_btn = gr.Button("🎙️ Start Recording", variant="primary", size="lg",visible=True)
                    stop_btn = gr.Button("⏹️ Stop Recording", variant="stop", size="lg",visible=True)

                status_display = gr.Textbox(
                    value="Ready to record your meeting",
                    label="Status",
                    interactive=False
                )

                with gr.Row():
                    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

                    # Construct the dynamic path to the GIF
                    record_gif_path = os.path.join(BASE_DIR, "assets", "record.gif")

                    record_indicator = gr.Image(
                        record_gif_path, label="Recording in progress",
                        visible=False,
                        show_download_button=False,
                        container=False,
                        height=100,
                        width=100)

        with gr.Column(scale=2):
            gr.Markdown("### Processing Status")
            with gr.Accordion("How to use", open=False):
                gr.Markdown("""
                1. Click "Start Recording" to begin capturing audio
                2. Records Both Side of the Meeting(with Noise Reduction)
                3. Click "Stop Recording" when finished
                4. View the transcription and summary in their respective tabs
                5. Download the complete notes as a PDF file

                6.If using BLUETOOTH will record only MIC

                7.Optimized to Use both GPU & CPU

                8.Records 10-15 Mins at a time without ERROR
                """)

            progress = gr.Progress(track_tqdm=True)

    with gr.Tabs() as tabs:
        with gr.TabItem("✍️ Transcription", id="transcription"):
            transcript_output = gr.Textbox(
                label="Full Meeting Transcription",
                placeholder="Transcription will appear here after recording...",
                interactive=False,
                lines=15
            )

        with gr.TabItem("📝 Summary", id="summary"):
            summary_output = gr.Textbox(
                label="Meeting Summary",
                placeholder="AI-generated summary will appear here after recording...",
                interactive=False,
                lines=10
            )

    with gr.Row():
        download_btn = gr.Button("📥 Download PDF Report", variant="secondary", size="lg")
        download_file = gr.File(label="Download Complete Meeting Notes", visible=True)

    with gr.Accordion("About ", open=False):
        gr.Markdown("""
        **MADE BY -JUBHAJIT DEB** (AI Minutes of Meeting) uses advanced AI to transcribe and summarize your meetings.

        **Features:**
        - Records Meeting on Both Side(Skype, Zoom , Meets , Teams , Whatsapp call)
        - Accurate speech-to-text transcription using Whisper AI
        - Intelligent summarization with BART
        - Multi-language support(en & hi)
        - Complete PDF report generation


        This tool helps you focus on your meeting while AI takes notes for you!
        """)

    # Event handlers
    start_btn.click(
        start_recording,
        [],
        [status_display, record_indicator, start_btn, stop_btn]
    )

    stop_btn.click(
        stop_recording,
        [],
        [status_display, record_indicator, stop_btn, start_btn]
    )

    stop_btn.click(
        process_audio,
        [],
        [transcript_output, summary_output]
    )

    download_btn.click(
        download_pdf,
        [],
        [download_file]
    )

if __name__ == "__main__":
    app.queue().launch()