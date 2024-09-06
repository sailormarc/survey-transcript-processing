# Introduction
The code in this project manipulates a dataset consisting of open-ended interviews, conducted in English and recorded through the in-house call center infrastructure of a partner company. 
The data have been collected via qualitative phone surveys with open-ended questions over two (pilot) experimental rounds. The survey questionnaire contains 7-9 question (depending on the client), with minor modifications between the two rounds. 

This readme file describes the pipeline in place to transcribe the interviews using speech-to-text models, split the transcriptions into blocks containing the distinct questions, performing text embedding and creating datasets for visualization and analysis.

**Note**: this is a prototype rather than a final product, so while rather clean, the code structure is not quite optimized.

## Requirements
The libraries needed to run the python code are in requirements.txt. These are mostly standard data science libraries, plus 
1.	openai
2.	deepgram-sdk
3.	pydub (for audio processing)
These can be installed via pip (and will be installed by running `pip install -r requirements.txt` from shell). However, pydub depends on ffmpeg which can’t be installed via pip, see https://github.com/jiaaro/pydub#installation for an installation guide. 

It is recommended to set up a virtual environment, e.g. with conda: 

```bash
conda create --name voicesofghana python=3.12
conda activate voicesofghana
pip install -r requirements.txt
```

In addition, you will need OpenAI, Deepgram and VoyageAI API keys. These are imported from a .env file by the config.py file in the code folder. Make sure to create a .env file with the text
```
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
DEEPGRAM_API_KEY = "YOUR_DEEPGRAM_API_KEY"
VOYAGE_API_KEY = "YOUR_VOYAGE_API_KEY"
```
and change the `dotenv_path` variable in config.py to wherever you store the .env file (not on dropbox cause you don’t want it to be synced). 

## Cleaning the audio files
With the raw files in `../../data/audio`, run the script `process_raw_audios.py` with positional arguments <directory> = path of folder containing the raw audios, <dta_path> = path to cleaned qualtrics data (from the survey) in dta format, and optional argument <min_duration> = the minimum duration for a recording to be kept (defaults to 1 min). The script does the following:
0.	extract the audio files from the correct agents 
1.	filters phone numbers by eligibility criteria (mostly consent question from the questionnaire), and remove audios of non-eligible numbers
2.	removes recordings below the minimum duration
3.	converts to .wav
4.	concatenates calls to the same phone number, in chronological order
5.	moves all resulting audio files to the voicefiles folder. Final files are called by the phone number of the customer, and a few folder are left with phone numbers with multiple recordings. These folders are a byproduct of the script and can be removed. 

**Note**: if you want to keep a copy of the raw audios, make a copy first because the script that processes the folder with the audios is in-place.

## Transcription
For the first round only, we obtained high-quality, human-made transcriptions, which we use as ground-truth to develop and test an automatic transcription pipeline. We tested several SOTA speech-to-text models and denoising procedures and landed on the deepgram-nova2 model. The transcriptions are postprocessed by an LLM of the GPT-4 family to improve speaker labeling and identify only 2 speakers. 

The code for transcribing and relabeling is in `transcribe_and_relabel.ipynb`. It takes as input audio files as processed by the script `process_raw_audios.py`. Just run all the cells to transcribe the round 1 and round 2 audios. 

## Question splitting
We leverage models from the GPT-4 family to split each text into blocks corresponding to the open-ended questions Q1-Q9. First, run the stata script `split_transcriptions_round1.do` (this is a preprocessing step). To perform the splits for round1 ground truth, round 1 nova2, and round 2 nova2, run all cells in `split_questions.ipynb`. 

**Note**: The extra step for round 1 is there for “historical” reasons (the project was started in stata). For round 2 (and eventually subsequent rounds), this step should be done as is done in `split_questions.ipynb` for the round2 calls.
**Note**: it may take up to several hours to run the splitting code on the full dataset, so plan accordingly if you want to do that. 
**Note**: for different questionnaires you might need to produce new examples (the current code needs one example for winners and one example for non-winners). Zero-shot prompting for question splitting (no examples) was tested and performed quite poorly.

## Text embeddings
To have a first look at the quality and signal in the data, we do some visualizations and analysis based on text embeddings. We use text-embedding models from openai and voyageai, which should be trained to map “semantically similar” texts to similar (in L^2 or cosine similarity) vectors in the (~1500 dimensional) embedding space. 

The notebook `create_datasets_for_visualization_analysis.ipynb` contains the code to generate the embeddings from the split transcriptions, put everything into a dataframe and save in pkl format. The three datasets obtained this way are saved in `../../data/transcription_all/embeddings` and can be used for visualization and analysis.  
