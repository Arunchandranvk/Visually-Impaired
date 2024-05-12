import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import os
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from gtts import gTTS
import speech_recognition as sr

os.environ["OPENAI_API_KEY"] = "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"

# PDF processing and language model loading code
pdfreader = PdfReader(r"D:/Internship Luminar/Students Disabled/P3_Career_Chatbot/Career.pdf")
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

text_splitter = CharacterTextSplitter(
    separator="/n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)
texts = text_splitter.split_text(raw_text)

embeddings = OpenAIEmbeddings()
document_search = FAISS.from_texts(texts, embeddings)

chain = load_qa_chain(OpenAI(), chain_type="stuff")

def speak_text(text):
    tts = gTTS(text=text, lang="en")
    tts.save("bot_response.mp3")
    os.system("start bot_response.mp3")

def get_response():
    user_input = user_input_entry.get("1.0", tk.END)
    bot_response = ""

    if user_input.strip().lower() in ["hi", "hello", "hey", "hy", "hi ruby", "hello ruby", "hey ruby", "hy ruby"]:
        bot_response = "Hello, welcome to Luminar Career Guide. How can I assist you today!"
    elif user_input.strip().lower() in ["bye", "by", "bye ruby", "by ruby", "thank you", "thanks"]:
        bot_response = "bye, and have a good career ahead"
    else:
        question = user_input.strip()
        if len(question) < 4:
            bot_response = "Please enter a valid question!"
        else:
            docs = document_search.similarity_search(user_input)
            bot_response = chain.run(input_documents=docs, question=question)

    chat_window.config(state=tk.NORMAL)
    chat_window.insert(tk.END, "You: " + user_input + "/n", "user")
    chat_window.insert(tk.END, "Ruby: " + bot_response + "/n", "bot")
    chat_window.config(state=tk.DISABLED)
    user_input_entry.delete("1.0", tk.END)

    speak_text(bot_response)

def get_input_from_speech():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)

    try:
        user_input = recognizer.recognize_google(audio)
        print("You said:", user_input)
        user_input_entry.insert(tk.END, user_input)
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
    except sr.RequestError as e:
        print(f"Error connecting to Google API: {e}")

root = tk.Tk()
root.title("Career Chatbot")

style = ttk.Style()
style.theme_use("clam")  # You can experiment with other themes: 'clam', 'alt', 'default', etc.

notebook = ttk.Notebook(root)
notebook.pack(fill=tk.BOTH, expand=True)

# Create the home_frame
home_frame = ttk.Frame(notebook)
home_frame.pack(fill=tk.BOTH, expand=True)
notebook.add(home_frame, text="Careerbot")

image = Image.open(r"D:/Internship Luminar/Students Disabled/P3_Career_Chatbot/img.webp")
max_width = 300
max_height = 200
image.thumbnail((max_width, max_height), Image.LANCZOS)
photo = ImageTk.PhotoImage(image)
image_label = tk.Label(home_frame, image=photo)
image_label.pack(pady=20)

chat_window = scrolledtext.ScrolledText(home_frame, wrap=tk.WORD, state=tk.DISABLED, width=70, height=10, bd=0)
chat_window.tag_configure("user", foreground="red")
chat_window.tag_configure("bot", foreground="black")
chat_window.pack(pady=20)

user_input_entry = tk.Text(home_frame, height=3, width=50)
user_input_entry.pack()

ask_button = ttk.Button(home_frame, text="Ask", command=get_response)
ask_button.pack(pady=20)

speak_button = ttk.Button(home_frame, text="Speak", command=get_input_from_speech)
speak_button.pack(pady=20)

# Configure background color
# root.configure(background='#DBE7C9')
# home_frame.configure(background='#DBE7C9')
# chat_window.configure(bg='#DBE7C9')
# user_input_entry.configure(bg='#DBE7C9')

root.mainloop()
