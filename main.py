import torch
import sys
import glob
import logging
import multiprocessing
import os
import time
import wave
import pyaudio
import threading

import whisper
from multiprocessing import freeze_support

from PyQt5.QtGui import QTextCharFormat, QTextCursor , QKeySequence

import configparser
import numpy as np
from networkx import edge_load_centrality, non_neighbors

config = configparser.ConfigParser()

from silero_vad import read_audio, get_speech_timestamps, load_silero_vad

from silero_vad.utils_vad import init_jit_model

config.read('config.ini')
# Parameters
samplerate = 44100  # Sample rate in Hertz

# Audio parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
SEGMENT_DURATION = int(config.get('appsettings', 'SEGMENT_DURATION'))  # Duration of each segment in seconds
REPEAT_COUNT = int(config.get('appsettings', 'REPEAT_COUNT'))  # Number of 5-second segments to record
FONT = config.get('appsettings', 'FONT')
FONT_SIZE = config.get('appsettings', 'FONT_SIZE')
MODEL_NAME = 'small.pt'  # config.get('appsettings', 'MODEL_NAME')
PYQT_EMIT_SLEEP_DELAY = float(config.get('appsettings', 'PYQT_EMIT_SLEEP_DELAY'))
experimental_mobile_cursor_mode = int(config.get('appsettings', 'experimental_mobile_cursor_mode'))
shortcut_start_transcribe = config.get('appsettings', 'SHORTCUT_START_TRANSCRIBE')
shortcut_stop_transcribe = config.get('appsettings', 'SHORTCUT_STOP_TRANSCRIBE')
shortcut_pause_record = config.get('appsettings', 'SHORTCUT_PAUSE_RECORD')

is_recording = None

OUTPUT_FILENAME_TEMPLATE = "segment_{}.wav"

result_sst_response = []

start_stt = False

LOG_PATH = os.path.join(os.getcwd(), 'whisper_logs')
os.makedirs(LOG_PATH, exist_ok=True)
os.makedirs(os.path.join(os.getcwd(), 'temp'), exist_ok=True)
log_file = os.path.join(LOG_PATH, 'whisper_logs.log')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QPushButton, QVBoxLayout,
    QHBoxLayout, QWidget, QAction, QMenuBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal , QEvent

from PyQt5.QtGui import QFont

recording_p = None
transcribe_p = None
model_load_event = None
is_paused = False
result_queue = None

c = 1


class RecordThread(QThread):
    update = pyqtSignal(str)  # Signal to emit when recording is done
    update_italics = pyqtSignal(str)

    def run(self):
        # Emit the signal to indicate that recording is done
        global result_sst_response, start_stt, recording_p, transcribe_p, MODEL_NAME, model_load_event, is_paused, result_queue, c, is_recording

        recording_p = None
        transcribe_p = None
        model_load_event = None
        is_paused = None
        is_recording = None
        result_queue = None

        try:

            queue = multiprocessing.Queue()
            result_queue = multiprocessing.Queue()

            model_load_event = multiprocessing.Event()

            is_paused = multiprocessing.Value('b', False)
            is_recording = multiprocessing.Value('b', True)

            # large chuck saving could take time : have another process to save accumulating chunks so save time can be avoided [Note]
            transcribe_p = multiprocessing.Process(target=transcribe_process,
                                                   args=(
                                                       queue, result_queue, model_load_event, MODEL_NAME,
                                                       is_recording,))
            transcribe_p = transcribe_p

            recording_p = multiprocessing.Process(target=recording_process,
                                                  args=(queue, model_load_event, is_paused, is_recording,))
            recording_p = recording_p
            logging.info("2 process started")
            transcribe_p.start()
            recording_p.start()

        except Exception as e:
            logging.error("error while record : " + str(e))

        self.update_italics.emit("Model Loading Please Wait  ...")

        model_load_event.wait()

        logging.info("Recording started")
        self.update_italics.emit("Model Loaded Recording Started ...")

        c = 1
        try:

            while is_recording.value:
                if not result_queue.empty():
                    print('o/p:')
                    if c % REPEAT_COUNT == 0:
                        self.update.emit(result_queue.get())
                        c += 1
                        continue

                    self.update_italics.emit(result_queue.get())
                    c += 1
                time.sleep(PYQT_EMIT_SLEEP_DELAY)
        except Exception as e:
            print("record_audio() exception : " + str(e))
            logging.error(f"Error in record_audio(): {str(e)}")

        recording_p.join()
        transcribe_p.join()

        recording_p = None
        transcribe_p = None

        try:
            temp_dir = os.path.join(os.getcwd(), "temp")
            files = glob.glob(os.path.join(temp_dir, "*"))
        except Exception as e:
            logging.error("Error getting files to delete" + str(e))


        for file in files:
            try:
                os.remove(file)
            except Exception as e:
                logging.error("Error while removing files : "+str(e))
                continue

        self.update_italics.emit("enable start button")

        print('recording ended !')


class TextEditor(QMainWindow):
    def __init__(self):
        global shortcut_start_transcribe , shortcut_stop_transcribe , shortcut_pause_record
        super().__init__()

        self.setWindowTitle("Speak2Write")
        self.setGeometry(100, 100, 600, 400)

        # Text editor
        self.text_edit = QTextEdit(self)
        self.text_edit.setStyleSheet(f"background-color: #2b2b2b; color: #dcdcdc; font-size: {FONT_SIZE}px;")

        self.font = QFont(FONT, int(FONT_SIZE))
        self.font.setPointSize(int(FONT_SIZE))
        self.text_edit.setFont(self.font)

        # Modern buttons
        self.start_button = QPushButton("Start Transcription", self)
        self.pause_button = QPushButton("Pause Recording", self)
        self.stop_button = QPushButton("Stop Transcription", self)

        # Set modern button styles
        button_style = """
            QPushButton {
                background-color: #3c3f41;
                color: #dcdcdc;
                font-size: 24px;
                border: 2px solid #4f5254;
                border-radius: 8px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #4f5254;
            }
            QPushButton:pressed {
                background-color: #5a5e62;
            }
            QPushButton:disabled {
                background-color: #2b2b2b;
                color: #757575;
            }
        """
        self.start_button.setStyleSheet(button_style)
        self.stop_button.setStyleSheet(button_style)
        self.pause_button.setStyleSheet(button_style)

        # Disable Stop button initially
        self.stop_button.setEnabled(False)
        self.pause_button.setEnabled(False)

        # Button actions
        self.start_button.clicked.connect(self.start_recording)
        self.stop_button.clicked.connect(self.stop_recording)
        self.pause_button.clicked.connect(self.pause_recording)

        # Layouts
        main_layout = QVBoxLayout()
        button_layout = QHBoxLayout()

        # Add buttons to layouts
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.stop_button)

        main_layout.addWidget(self.text_edit)
        main_layout.addLayout(button_layout)

        # Main widget
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Dark mode
        self.setStyleSheet("background-color: #1e1e1e; color: #dcdcdc;")

        self.fs = 44100  # Sample rate

        # RecordThread for long-running task
        self.record_thread = RecordThread(self)
        self.record_thread.update.connect(self.update_text)
        self.record_thread.update_italics.connect(self.insert_text_in_italics)

        self.cursor = self.text_edit.textCursor()
        self.cursor_start_position = self.cursor.position()
        self.cursor_end_position = self.cursor.position()

        # shortcut : start transcribe
        shortcut_start_transcribe = shortcut_start_transcribe.replace("'", "")
        self.create_shortcut(QKeySequence(f"Ctrl+{shortcut_start_transcribe}"), self.start_button)
        # shortcut : pause transcribe
        shortcut_pause_record = shortcut_pause_record.replace("'", "")
        self.create_shortcut(QKeySequence(f"Ctrl+{shortcut_pause_record}"), self.pause_button)
        # shortcut : stop recording
        shortcut_stop_transcribe = shortcut_stop_transcribe.replace("'", "")
        self.create_shortcut(QKeySequence(f"Ctrl+{shortcut_stop_transcribe}"), self.stop_button)

    def start_recording(self):

        global recording_p, transcribe_p

        if recording_p is not None:
            recording_p.terminate()
            logging.info("recording process : force closed")
        if transcribe_p is not None:
            transcribe_p.terminate()
            logging.info("transcribe process : force closed")

        try:
            global start_stt
            cursor = self.text_edit.textCursor()
            cursor.movePosition(QTextCursor.End)

            self.cursor_start_position = cursor.position()
            self.cursor_end_position = cursor.position()

            start_stt = True
            # Start recording logic here
            self.start_button.setEnabled(False)
            self.pause_button.setEnabled(True)
            if not experimental_mobile_cursor_mode:
                self.text_edit.setTextInteractionFlags(Qt.NoTextInteraction)

            temp_dir = os.path.join(os.getcwd(), "temp")
            files = glob.glob(os.path.join(temp_dir, "*"))
        except Exception as e:
            logging.error("Error in start_recording:" + str(e))

        for file in files:
            try:
                os.remove(file)
            except Exception as e:
                logging.error("Error while removing files")
                continue

        self.record_thread.start()

    def pause_recording(self):
        global is_paused, c
        if self.pause_button.text() == "Pause Recording":
            self.text_edit.setTextInteractionFlags(Qt.TextEditorInteraction)

            cursor = self.text_edit.textCursor()
            format_ = QTextCharFormat()
            format_.setFontItalic(False)
            cursor.insertText(". ", format_)

            is_paused.value = True
            self.pause_button.setText("Continue Recording")

        else:
            self.pause_button.setText("Pause Recording")
            c = 1
            self.text_edit.setTextInteractionFlags(Qt.NoTextInteraction)

            cursor = self.text_edit.textCursor()
            cursor.movePosition(QTextCursor.End)

            self.cursor_start_position = cursor.position()
            self.cursor_end_position = cursor.position()

            is_paused.value = False
            print('continued recording ...')

    def update_text(self, response):
        # eliminate last italics text
        if self.cursor_start_position != self.cursor_end_position:
            self.remove_text_between_positions(self.cursor_start_position, self.cursor_end_position)

        cursor = self.text_edit.textCursor()

        format_ = QTextCharFormat()
        format_.setFontItalic(False)

        self.cursor_start_position = cursor.position()

        cursor.setPosition(self.cursor_start_position)
        cursor.insertText(response + " ", format_)

        # update latest start position and end position after writing
        cursor.movePosition(QTextCursor.End)
        self.cursor_start_position = cursor.position()
        self.cursor_end_position = cursor.position()

    def insert_text_in_italics(self, response):
        # Get the text cursor
        if response == "Model Loaded Recording Started ...":
            self.stop_button.setEnabled(True)

        if response == "enable start button":
            self.start_button.setEnabled(True)
            return

        if self.cursor_start_position != self.cursor_end_position:
            self.remove_text_between_positions(self.cursor_start_position, self.cursor_end_position)

        cursor = self.text_edit.textCursor()

        # setting position where to write :
        cursor.setPosition(self.cursor_start_position)

        # Create a QTextCharFormat object and set it to italics
        format_ = QTextCharFormat()
        format_.setFontItalic(True)
        cursor.insertText(" :: " + response, format_)

        cursor.movePosition(QTextCursor.End)
        self.cursor_end_position = cursor.position()

    def remove_text_between_positions(self, start_pos, end_pos):
        # Get the QTextCursor from the QTextEdit
        cursor = self.text_edit.textCursor()

        # Set the cursor position to the start of the text you want to remove
        cursor.setPosition(start_pos)
        cursor.setPosition(end_pos, QTextCursor.KeepAnchor)  # Anchor the cursor to select the text

        # Remove the selected text
        cursor.removeSelectedText()

    def stop_recording(self):
        # Stop recording logic here
        global start_stt, recording_p, transcribe_p, is_paused, is_recording
        is_recording.value = False
        self.stop_button.setEnabled(False)
        # self.start_button.setEnabled(True)
        self.text_edit.setTextInteractionFlags(Qt.TextEditorInteraction)
        self.pause_button.setEnabled(False)

        cursor = self.text_edit.textCursor()
        format_ = QTextCharFormat()
        format_.setFontItalic(False)
        cursor.insertText(".", format_)

        if self.pause_button.text() == "Continue Recording":
            self.pause_button.setText("Pause Recording")
            is_paused.value = False

        start_stt = False

        # recording_p.terminate()
        transcribe_p.terminate()

        temp_dir = os.path.join(os.getcwd(), "temp")
        files = glob.glob(os.path.join(temp_dir, "*"))

        for file in files:
            try:
                os.remove(file)
            except Exception as e:
                logging.error(f"Error in stop_recording(): {str(e)}")
                continue

        logging.info("Recording stopped")

        # self.start_button.setEnabled(True)

    def create_shortcut(self, key_sequence, button):
        shortcut_action = QAction(self)
        shortcut_action.setShortcut(key_sequence)
        self.addAction(shortcut_action)
        shortcut_action.triggered.connect(button.click)



    def closeEvent(self,  event: QEvent):
        # Stop the worker thread if it's running
        global recording_p, transcribe_p

        if recording_p is not None:
            recording_p.terminate()
            logging.info("recording process : force closed")
        if transcribe_p is not None:
            transcribe_p.terminate()
            logging.info("transcribe process : force closed")

        event.accept()


class ContinuousAudioRecorder:
    def __init__(self, queue, n):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT, channels=CHANNELS,
                                  rate=RATE, input=True, frames_per_buffer=CHUNK)
        self.segment_count = n
        self.queue = queue

    def record_segment(self, duration, frames):
        num_chunks = int(duration * RATE / CHUNK)

        for _ in range(num_chunks):
            data = self.stream.read(CHUNK)
            frames.append(data)

        self._save_wav(frames)
        self.segment_count += 1

    def _save_wav(self, frames):
        t1 = time.time()
        filename = OUTPUT_FILENAME_TEMPLATE.format(self.segment_count)
        with wave.open("temp//" + filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
        t2 = time.time()
        print(f"Saved {filename} in: " + str(t2 - t1))

        self.queue.put(filename)

    def start_recording(self, duration, repeat_count):
        try:
            frames = []
            for _ in range(repeat_count):
                self.record_segment(duration, frames)
        finally:
            self.stop_recording()

    def stop_recording(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


def recording_process(queue, model_load_event, is_paused, is_recording, ):
    n = 1
    model_load_event.wait()

    import configparser
    config = configparser.ConfigParser()
    config.read('config.ini')
    record_sleep_delay = float(config.get('appsettings', 'RECORD_SLEEP_DELAY'))

    while is_recording.value:

        if is_paused.value:
            continue

        try:
            print('recording started ...')
            recorder = ContinuousAudioRecorder(queue, n)
            recorder.start_recording(SEGMENT_DURATION, REPEAT_COUNT)
            n = recorder.segment_count
            print('recording ended ... next chunk begins !')
            time.sleep(record_sleep_delay)
        except Exception as e:
            print("recording_process() exception : " + str(e))
            logging.error(f"Error in recording_process(): {str(e)}")
            print('exited !')

    logging.info("Recording Process ended !")


def transcribe_process(queue, result_queue, model_load_event, MODEL_NAME, is_recording, ):
    audio_model = whisper.load_model("models//" + MODEL_NAME.replace('"', ''),
                                     "cuda" if torch.cuda.is_available() else "cpu")
    vad_model = init_jit_model("models//" + "silero_vad.jit")
    print('model loaded !')
    model_load_event.set()

    import configparser
    config = configparser.ConfigParser()
    config.read('config.ini')
    threshold = int(config.get('appsettings', 'THRESHOLD'))

    # Define a threshold for energy to classify as speech or non-speech

    while is_recording.value:
        try:
            file = queue.get()
            if not os.path.isfile("temp//" + file):
                queue.put(file)
                continue

            try:

                wav = read_audio("temp//" + file)  # backend (sox, soundfile, or ffmpeg) required!
                speech_timestamps = get_speech_timestamps(wav, vad_model)

                if len(speech_timestamps) < threshold:
                    logging.info("Doesnt have speech removing : " + file)
                    print('Doesnt have speech removing ')
                    os.remove("temp//" + file)
                    result_queue.put("")
                    continue

                t1 = time.time()
                result = audio_model.transcribe("temp//" + file, language='en',  task='transcribe', fp16=True)
                t2 = time.time()
                print(f'time taken whisper {file}: ' + str(t2 - t1))
                logging.info(f'time taken whisper {file}: ' + str(t2 - t1))
                text = result['text'].strip()
                result_queue.put(text)

                print(f"transcribed: {file}")
                logging.info(f"transcribed: {file}")
                os.remove("temp//" + file)
            except Exception as e:
                print('some exception in transcription', str(e))
                logging.error(f"Error in transcribe_process(): {str(e)}")
                queue.put(file)
                continue
        except Exception as e:
            print('queue empty')
            logging.error(f"Error in transcribe_process(): {str(e)}")
            continue

    print('transcribe process ended !')
    logging.info("transcribe Process ended !")


if __name__ == "__main__":
    freeze_support()
    app = QApplication(sys.argv)
    editor = TextEditor()
    editor.show()
    sys.exit(app.exec_())

