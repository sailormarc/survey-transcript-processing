# This script contains most functions used in the project. They are divided into the following blocks: 

# Speech-to-text transcription
# Dataset handling
# Text file processing
# Question splitting 
# Text embeddings and visualization

# It also initializes the OpenAI API, Voyage API and Deepgram API clients, and tests the connection to the openAI API (if main).

# The script is meant to be imported in the main scripts and notebooks, where the functions are called.


# import config file from parent directory
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
import config

# import required libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import os
from tqdm import tqdm
import logging
import concurrent.futures
import time
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import signal

# Libraries for t-SNE visualization
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


# import OpenAI API and start the client
from openai import OpenAI
client = OpenAI()

# import voyage API and start the client
import voyageai
vo = voyageai.Client(timeout=500)

# import deepgram API and start the client
from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)
import gc
import httpx

# get DEEPGRAM_API_KEY from environment
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
deepgram: DeepgramClient = DeepgramClient(DEEPGRAM_API_KEY)



#####################################################################
######################### Transcription #############################
#####################################################################

# Function to remove line breaks that originate from Deepgrams Paragraph output
def remove_line_breaks(transcript):
    """
    Removes line breaks from a transcript and formats it by grouping speech lines under their respective speakers.

    Args:
        transcript (str): The transcript containing line breaks.

    Returns:
        str: The formatted transcript with speech lines grouped under their respective speakers.
    """
    new_transcript = []
    current_speaker = None
    for line in transcript.split('\n'):
        if line.startswith("Speaker "):
            if current_speaker is not None:
                new_transcript.append("")  # Add line break between speakers
            current_speaker = line.strip()  # Set current speaker
            new_transcript.append(current_speaker)  # Add speaker line
        elif current_speaker is not None:  # Check if current_speaker is set
            new_transcript[-1] += " " + line.strip()  # Append speech to the last speaker
    return new_transcript

# Function for transcription using Deepgram
def nova2(input_folder, output_folder):
    """
    Transcribes audio files in the input folder using the Nova-2 Phone Call model and saves the transcriptions as text files in the output folder.

    Args:
        input_folder (str): The path to the folder containing the audio files to be transcribed.
        output_folder (str): The path to the folder where the transcriptions will be saved.

    Returns:
        None
    """
    print(DEEPGRAM_API_KEY)
    for root, _, files in os.walk(input_folder):

        # Determine the corresponding output subfolder
        output_subfolder = os.path.join(output_folder) # for subfolders add "transcripts_" + os.path.relpath(root, input_folder))

        # Create the output subfolder if it doesn't exist
        os.makedirs(output_subfolder, exist_ok=True)

        for filename in files:
            if filename.endswith(".wav"):
                input_file = os.path.join(root, filename)
                output_file = os.path.join(output_subfolder, os.path.splitext(filename)[0] + ".txt")

                print("Processing file:", input_file)

                # Check if transcript file already exists
                if not os.path.exists(output_file):
                    try:
                        with open(input_file, "rb") as file:
                            buffer_data = file.read()

                        payload = {
                            "buffer": buffer_data
                        }

                        options = {
                            "model": "nova-2-phonecall",
                            "language": "en",
                            "smart_format": True,
                            "punctuate": True,
                            "paragraphs": True,
                            "diarize": True,
                            "keywords": "cedi" "Umoja"
                        }

                        response = deepgram.listen.prerecorded.v("1").transcribe_file(
                            payload, options, timeout=httpx.Timeout(300.0, connect=10.0)
                        )

                        transcript = response.results.channels[0].alternatives[0].paragraphs.transcript

                        # Post-process the transcript to remove line breaks
                        new_transcript = remove_line_breaks(transcript)

                        # Write the modified transcript to the output file
                        with open(output_file, "w") as txt_file:
                            for line in new_transcript:
                                txt_file.write(line + "\n")

                    finally:
                        gc.collect()

# Function to relabel speakers using OpenAI API
def relabel_speakers(transcript):
    """
    Relabels the speakers in a transcript using OpenAI API.

    Args:
        transcript (str): The transcript to be relabeled.

    Returns:
        str: The relabeled transcript.
    """
    system_prompt1 = """Given the transcript provided, please relabel the speakers as either 'caller' or 'agent' based on their dialogue characteristics.
    Agents are identified by the fact that they introduce themselves at the beginning, that they explain why they called and that they ask questions to the client.
    Clients are identified by the fact they answer questions and talk about their personal experience and their lives.
    If there are more than two speakers assign those either to 'caller' or 'agent' using the same logic as above. Please return the entire relabeled transcript!"""

    system_prompt2 = """Given the transcript provided, which is from a callcenter survey, please relabel the speakers as either 'caller' or 'agent' based on their dialogue characteristics.
    Here's a breakdown of how to identify each speaker, it is very important to accurately label them and be consistent (dont switch from labeling Speaker 1 as "caller" to labeling them as "agent"):
    Agent: - Introduces themselves at the beginning of the conversation with statements like "Good morning, I'm calling from ..."
    - Asks questions following a questionnaire and guides the conversation.
    - Explains the purpose of the call which is a survey for research purposes.
    Caller:
    - Responds to questions asked by the agent.
    - Shares personal experiences and opinions with the agent
    If there are more than two speakers in the conversation, apply the same logic as above to determine whether they should be labeled as 'caller' or 'agent' based on their dialogue characteristics.
    Please ensure that the relabeled transcript maintains the flow of the conversation and accurately represents the interactions between the speakers.
    Return the entire relabeled transcript with the updated speaker designations. Avoid labels like "Caller 1". The only acceptable labels are "Caller" and "Agent". """

    user_prompt = f"{transcript}"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt2},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.05,
        max_tokens=4095,
        top_p=1
    )

    return response.choices[0].message.content

# Function to relabel speakers in a folder of transcripts
def run_relabeling(input_folder, output_folder):
    """
    Relabels the speakers in a folder of transcripts.

    Args:
        input_folder (str): The path to the folder containing the transcripts to be relabeled.
        output_folder (str): The path to the folder where the relabeled transcripts will be saved.

    Returns:
        None
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):

        if filename.endswith(".txt"):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, os.path.splitext(filename)[0] + ".txt")

            print("Processing file:", input_file)

            if not os.path.exists(output_file):

                # Load ASR-generated transcript from file
                with open(input_file, "r", encoding="utf-8") as file:
                    transcript = file.read()

                    # Relabel speakers for part 1
                    relabeled_transcript = relabel_speakers(transcript)

                    # Save relabeled transcript
                    with open(output_file, "w", encoding="utf-8") as output:
                        output.write(relabeled_transcript)

                    print("Result saved to:", output_file)



#####################################################################
######################### Dataset handling ##########################
#####################################################################

# for question splitting

def reshape_answers_split_long(df, embedded_columns, num_questions=7):
    """
    Processes the dataset with question-answer columns and filters based on correct_num_seg.

    Args:
        df (pandas.DataFrame): Pandas dataframe containing the data.
        embedded_columns (list): List of column names containing the question-answer embeddings.
        num_questions (int, optional): Number of questions in the dataset. Defaults to 7. Should be set to 9 for round 2.

    Returns:
        pandas.DataFrame: A new pandas dataframe in long format with filtered data.

    """

    # Filter rows with correct_num_seg True
    df_filtered = df[df["correct_num_seg"] == True]

    # Create a long format dataframe
    id_vars=["reference_phone", "winner_label"]+[f"question_id_{i+1}" for i in range(num_questions)]
    df_long = df_filtered.melt(id_vars, value_vars=embedded_columns, var_name = 'question_number', value_name = 'question_answer_emb')
    df_long['question_number'] = df_long['question_number'].apply(lambda x: int(x.split("_")[2]))
    df_long['question_id'] = df_long.apply(lambda row: row[f"question_id_{int(row['question_number'])}"], axis=1)

    # Filter out rows with None in question_answer
    df_long = df_long.dropna(subset=["question_answer_emb"])

    return df_long

def reshape_dataset(df, pivot_index = 'reference_phone', file_level_vars = ['winner', 'full_transcription'], pivot_col = 'question_num', pivot_values = ['question_id', 'question_text', 'question_answer']):

    # Create a separate DataFrame with 'file_name', 'winner', and 'transcription'
    info_df = df[[pivot_index] + file_level_vars].drop_duplicates()

    # Pivot the original DataFrame for 'question_id'
    df_pivot = df.pivot(index=pivot_index, columns=pivot_col, values=pivot_values)

    # Flatten columns
    df_pivot.columns = ['_'.join((col[0], str(col[1]))) for col in df_pivot.columns] 

    # Merge the info DataFrame with the pivoted DataFrame
    df_wide = info_df.merge(df_pivot, on=pivot_index)

    # add a column with the file name = reference phone as string
    df_wide['file_name'] = df_wide[pivot_index].astype(str)

    # Replace Nas
    df_wide = df_wide.fillna("")
    return df_wide

def prepare_dataset_for_split(data_path, num_quest_winner, num_quest_loser, question_map_loser, question_map_winner):
  """
  This function cleans and prepares a Voices of Ghana dataset for export.

  Args:
      data_path (str): Path to the excel file with transcripts.
      output_path (str): Path to save the cleaned excel file.
      day (int): Day for the output filename.
      month (str): Month for the output filename.
  """

  # Read excel data
  df = pd.read_excel(data_path)

  # Expand data based on winner
  df_winner_0 = df.loc[df['winner'] == 0].copy()
  df_winner_1 = df.loc[df['winner'] == 1].copy()
  df = pd.concat([df_winner_0 for _ in range(num_quest_loser)] + [df_winner_1 for _ in range(num_quest_winner)], ignore_index=True)
  df = df.sort_values(by=['winner', 'reference_phone'])
  df = df.reset_index(drop=True) 

  # Add new variables
  df['question_id'] = "self_intro"
  df['question_text'] = "To begin, we would like to know you better. Can you tell me a little bit more about yourself and your family?"
  df['question_answer'] = ""
  df.rename(columns={'transcription': 'full_transcription'}, inplace=True)

  # Fill question details based on winner and order
  df['question_num'] = df.groupby('reference_phone').transform('cumcount')
  
  # Fill question details
  df[['question_id', 'question_text']] = df.apply(fill_question_details, axis=1, args=(question_map_loser, question_map_winner))

  return df

# miscellaneous

def merge_datasets(simple_transcripts, dta, columns_to_keep = ['file_name', 'winner', 'transcription']):
    dta = dta[columns_to_keep]
    dta = dta.rename(columns = { 'file_name' : 'filename', 'transcription' : 'ground_truth'})
    dta['filename'] = dta['filename'].map(clean_filename)
    return pd.merge(dta, simple_transcripts, how='right', on='filename')



#####################################################################
######################## Text file processing #######################
#####################################################################

def load_text_files(folder_path, filenames = None):
    """Load text files that are listed in the filenames list.
    
    Args:
        folder_path (str): Path to the folder containing text files.
        filenames (list): List of filenames to include (without .txt extension).
    
    Returns:
        pandas dataframe with columns: filename, transcription.
    """
    data = []

    if filenames is None:
        filenames_with_extension = []
        for file in os.listdir(folder_path):
            if file.endswith(".txt"):
                filenames_with_extension.append(file)
    else: 
        # Add .txt extension for comparison
        filenames_with_extension = [name + ".txt" for name in filenames]
    
    # Loop through each file in the directory
    for file in os.listdir(folder_path):
        if file in filenames_with_extension:
            # Construct full file path
            file_path = os.path.join(folder_path, file)
            # Open and read the content of the file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                data.append((file[:-4], content))  # Remove the .txt extension from filename
        else:
            print(file+" not in file list.\n")
    
    df = pd.DataFrame(data, columns=['reference_phone', 'transcription'])
    # convert reference_phone to int
    df['reference_phone'] = df['reference_phone'].astype(int)
    return df

def clean_filename(filename):
    # Remove "Audio name:", "Audio:", and ".mp3" from the filename
    filename = filename.replace("Audio name", "").replace("Audio", "").replace("AUDIO NAME:", "").replace("Audio", "").replace(":", "").replace(".mp3", "").replace(".mp", "")
    if "-all" not in filename:
        filename += "-all"
    return filename.strip()

def read_transcription_from_phone_number(phone_number, directory):
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            if phone_number in filename:
                with open(os.path.join(directory, filename), "r") as f:
                    return f.read()
    return None

def filter_by_word_count(text, min_words):
    return (len(text.split()) >= min_words) if pd.notna(text) else False

def make_dataset_from_txt_and_dta(dta_path, output_path, directory, min_words = 500):
  """
  Processes the dataset and saves it as an excel file.

  Args:
      dta_path: Path to the (cleaned) dta file.
      output_path: Path to save the processed data as an excel file.
      directory: Path to directory with txt files
      min_words: Minimum word count for a transcription to be kept (default: 500).


  Returns:
      A tuple containing the processed pandas dataframe and the number of observations left after filtering.
  """

  # Read the dta file
  df = pd.read_stata(dta_path)

  # Add a 'transcription' column with content from corresponding txt files (if available)
  df["transcription"] = df["reference_phone"].apply(
      lambda phone_number: read_transcription_from_phone_number(str(int(phone_number)), directory)
      if pd.notna(phone_number) else None  # Handle missing phone numbers with None
  )

  # Filter for English language, drop NaN values in 'transcription', and handle missing reference_phone
  df_filtered = df[(df["language"] == "English") & df["transcription"].notna()]

  df_filtered = df_filtered.dropna(subset=["reference_phone"])  # Drop rows with missing reference_phone
 
  df_filtered = df_filtered[df_filtered["transcription"].apply(filter_by_word_count, min_words=min_words)]
  
  # Save the dataframe as excel
  df_filtered.to_excel(output_path, index=False)

  # Return the filtered dataframe and number of observations left
  return df_filtered, len(df_filtered)

def cosine_similarity(v1, v2):
    """
    Calculate the cosine similarity between two vectors.

    Args:
        v1 (numpy array): First vector.
        v2 (numpy array): Second vector.

    Returns:
        Cosine similarity between the two vectors.
    """
    # make sure to handle cases where v1 or v2 is a zero vector
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))



#####################################################################
####################### Question Splitting ##########################
#####################################################################

# define system prompt to give task instruction to the language model 
SYSTEM_PROMPT_SPLIT = 'You are a highly skilled AI trained in language comprehension. I will present you with a series of texts containing interviews between an Agent (A) and a Caller (C). In each text, the Agent is surveying the Caller with a series of N open-ended questions. Each question is followed by an answer and possibly further discussion.\n\
    For each interview, I will provide you with the sequence of blocks and the corresponding questions. Your task is to split the text into blocks corresponding to the different questions. More precisely, block i should start with the Agent asking question i, and end right before the Agent asks question i+1. Note that this is a real conversation, and the agent might use small variations of the questions I will provide you.\
    Your output will have the following structure that uses XML delimiters:\
    \n\n\
    <STEP1> Here you store the beginning and end of each block as: \n Beginning of block 1: ... End of block 1: ... \n ... \n Beginning of block N: ... End of block N. \n </STEP1>\
    \n\n\
    <STEP2> In the preamble the Agent asks for consent and some demographic information such as age, employment status, and education level. This is is everything that comes before the Agent starts asking questions from the questionnaire.\n\
    The conclusion consists of the final greetings (after the last question has been answered and the interview is over). \n\
    Last sentence of the preamble: store it here.\n\
    First sentence of the conclusion: store it here.\n\
    </STEP2>\
    \n\n\
    <REASONING> Reason on the following: 1. Is the number of blocks you detected equal to the number of blocks you were supposed to find? If not, correct possible mistakes here. \n\
    2. Does each block end where the next block begins? If not, correct possible mistakes here.\n\
    </REASONING>\
    \n\n\
    <BLOCK1> Here you copy the block of text which starts with the beginning of block 1 and ends at the end block 1 </BLOCK1>\
    \n\n\
    ...\
    \n\n\
    <BLOCKN> Here you copy the block of text which starts with the beginning of block N and ends at the end block N </BLOCKN>\
    \n\n\
    For STEP1, keep in mind that each block begins with the agent asking the corresponding question, and ends right before the Agent asks the next question.\
    To populate the fields <BLOCK1> to <BLOCKN>, use the points you found in STEP1 to split the text, considering and correcting possible mistakes as you found in REASONING. Ignore the preamble and the conclusion that you found in STEP2.\n\
    OF UTMOST IMPORTANCE: each <BLOCK> should be populated with **all** the text from the interview between the corresponding beginning and end points you found in STEP1.'


def construct_segments(row, num_losers=6, num_winners=7):
    # This list will store the dictionaries for each question id and text
    questions = []
    
    # Set question number range
    n_questions = (num_losers if row['winner'] == 0 else num_winners)
    # Iterate over each column that has a question_id tag in the row
    for i in range(n_questions):  # Adjust the range according to the number of questions
        question_id_key = f'question_id_{i+1}'
        question_text_key = f'question_text_{i+1}'
        
        # Check if the keys are actually in the DataFrame's columns
        if question_id_key in row and question_text_key in row:
            # Get the question text, replace empty string with None
            question_text = row[question_text_key]
            
            # Create the dictionary for this question
            question_dict = {
                'topic': row[question_id_key],
                'question_text': question_text
            }
            
            # Append the dictionary to the list
            questions.append(question_dict)
    
    return questions

def construct_user_prompt(segments_dict, transcription):
    """
    Constructs a user prompt by combining the segments' information and the transcription.

    Args:
        segments_dict (dict): A list of dictionaries containing information about each segment.
            Each segment should have the following keys:
                - 'topic': The topic of the segment.
                - 'question_text': The question text for the segment (optional).

        transcription (str): The transcription text.

    Returns:
        str: The constructed user prompt.

    Example:
        segments_dict = [
            {'topic': 'Introduction', 'question_text': 'Please introduce yourself.'},
            {'topic': 'Experience', 'question_text': 'Tell us about your previous work experience.'},
            {'topic': 'Skills', 'question_text': 'What are your key skills?'}
        ]
        transcription = "Hello, my name is John Doe. I have 5 years of experience in software development."

        prompt = construct_user_prompt(segments_dict, transcription)
        print(prompt)
        # Output:
        # Block 1: topic = Introduction, question text = Please introduce yourself.
        # Block 2: topic = Experience, question text = Tell us about your previous work experience.
        # Block 3: topic = Skills, question text = What are your key skills?
        #
        # The text begins on the following line.
        #
        # ""Hello, my name is John Doe. I have 5 years of experience in software development.""
    """

    segments_text = "\n".join( [ f'Block {i+1}: ' + f"topic = {segments_dict[i]['topic']}" + 
        (f", question text = {segments_dict[i]['question_text']}." if segments_dict[i]['question_text'] != "" else '.') 
        for i in range(len(segments_dict)) ] )
    
    prompt = segments_text + '\nThe text begins on the following line.\n\n"""' + transcription + '"""'
    return prompt

def produce_example(row, num_loser_questions=6, num_winner_questions=7):
    """
    Constructs an example consisting of user prompt and exmple completion, based on the given row data.

    Args:
        row (dict): A dictionary containing the row data.
        num_loser_questions (int, optional): The number of questions asked to non-winners. Defaults to 6.
        num_winner_questions (int, optional): The number questions asked to winners. Defaults to 7.

    Returns:
        tuple: A tuple containing the constructed prompt and the example blocks (completion).

    Example:
        >>> row = {'full_transcription': 'Lorem ipsum dolor sit amet', 'winner': 1, 'step1': 'Step 1', 'step2': 'Step 2', 'reasoning': 'Reasoning', 'question_answer_1': 'Question 1', 'question_answer_2': 'Question 2'}
        >>> produce_example(row)
        ('Constructed prompt', 'Step 1\n\nStep 2\n\nReasoning\n\n<BLOCK1>\nQuestion 1\n</BLOCK1>\n\n<BLOCK2>\nQuestion 2\n</BLOCK2>')
    """
    
    segments = construct_segments(row)
    prompt = construct_user_prompt(segments, row['full_transcription'])
    num_questions = num_loser_questions if row['winner'] == 0 else num_winner_questions
    blocks = '\n\n'.join([(f'<BLOCK{i+1}>\n' + row[f'question_answer_{i+1}'] + f'\n</BLOCK{i+1}>') for i in range(num_questions)])
    return prompt, '\n\n'.join((row['step1'], row['step2'], row['reasoning'], blocks))

def split_questions_0shot(system_prompt, segments_dict, transcription, model = "gpt-4o", temperature = 0.05, max_tokens = 4000):
    """
    Split questions using OpenAI's language models.

    Args:
        system_prompt (str): The system prompt to be used in the conversation.
        segments_dict (list): A list of dictionaries containing information on the different blocks of the interview.
        transcription (str): The transcription of the audio.
        model (str, optional): The name of the GPT model to use. Defaults to "gpt-4o".
        temperature (float, optional): The temperature parameter for text generation. Defaults to 0.05.
        max_tokens (int, optional): The maximum number of tokens in the generated text. Defaults to 4000.

    Returns:
        str: The generated text containing the split questions.
    """
        
    user_prompt = construct_user_prompt(segments_dict, transcription)

    completion = client.chat.completions.create(
    model=model,
    messages=[
        {
        "role": "system",
        "content": system_prompt
        },
        {
        "role": "user",
        "content": user_prompt 
        }
    ],
    temperature=temperature,
    max_tokens=max_tokens
    )

    # segments = completion.choices[0].message.content.split("|")
    # segments = [re.search(r'Segment [0-9]: (.*)', q).group(1) for q in segments]

    # num_segments = len(segments) == len(segments_dict)

    # return num_segments, segments 
    return completion.choices[0].message.content.strip()

def extract_blocks(text):
  """
  Extracts blocks delimited by <BLOCKi> </BLOCKi> tags from a string.

  Args:
      text: The string containing the blocks.

  Returns:
      A tuple containing a list of blocks and the number of blocks found.
  """
  blocks = []
  start = 0
  while True:
    open_tag = text.find("<BLOCK", start)
    if open_tag == -1:
      break
    close_tag = text.find("</BLOCK", open_tag)
    if close_tag == -1:
      raise ValueError("Mismatched tags in text")
    block_num = text[open_tag + len("<BLOCK"):open_tag + len("<BLOCK") + 1]
    if not block_num.isdigit():
      raise ValueError("Invalid block number format")
    blocks.append(text[open_tag + len("<BLOCK") + len(block_num) + 2:close_tag])
    start = close_tag + len("</BLOCK>")
  return blocks, len(blocks)

def split_questions_1shot(system_prompt, segments_dict, transcription, user_example, assistant_example, model = "gpt-4-turbo-2024-04-09", temperature = 0, max_tokens = 4000):
    """
    Split questions using OpenAI's chat-based language model.

    Args:
        system_prompt (str): The system prompt to provide context for the conversation.
        segments_dict (list): A list of dictionaries containing information on the different blocks of the interview.
        transcription (str): The transcription of the conversation.
        user_example (str): An example user message.
        assistant_example (str): An example assistant message.
        model (str, optional): The language model to use. Defaults to "gpt-4-turbo-2024-04-09".
        temperature (float, optional): The temperature parameter for controlling randomness in the model's output. Defaults to 0.
        max_tokens (int, optional): The maximum number of tokens in the model's response. Defaults to 4000.

    Returns:
        str: The generated completion from the language model.

    Raises:
        OpenAIException: If there is an error in the OpenAI API request.

    """
       
    user_prompt = construct_user_prompt(segments_dict, transcription)

    completion = client.chat.completions.create(
    model=model,
    messages=[
        {
        "role": "system",
        "content": system_prompt
        },
        {
        "role": "user",
        "content": user_example 
        },
        {
        "role": "assistant",
        "content": assistant_example 
        },
        {
        "role": "user",
        "content": user_prompt 
        }
    ],
    temperature=temperature,
    max_tokens=max_tokens
    )

    # segments = completion.choices[0].message.content.split("|")
    # segments = [re.search(r'Segment [0-9]: (.*)', q).group(1) for q in segments]

    # num_segments = len(segments) == len(segments_dict)

    # return num_segments, segments 
    return completion.choices[0].message.content.strip()

def count_similarity(texts):
    vectorizer = CountVectorizer(stop_words='english')
    counts = vectorizer.fit_transform(texts)
    a = (counts @ counts.T).A
    return a[0,1] / np.sqrt(a[0,0]*a[1,1]) if a[0,0]*a[1,1] != 0 else 0

def call_api_with_timeout(function, *args, timeout_seconds=360, retry_wait=5, **kwargs):
    """
    Calls the specified function with the given arguments and a timeout.

    Args:
        function (callable): The function to be called.
        *args: Variable length argument list to be passed to the function.
        timeout_seconds (int, optional): The maximum number of seconds to wait for the function to complete.
            Defaults to 360.
        retry_wait (int, optional): The number of seconds to wait before retrying the function in case of a timeout.
            Defaults to 5.
        **kwargs: Arbitrary keyword arguments to be passed to the function.

    Returns:
        The result of the function call.

    Raises:
        concurrent.futures.TimeoutError: If the function does not complete within the specified timeout.

    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(function, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            print("API call #1 timed out. Retrying after", retry_wait, "seconds...")
            time.sleep(retry_wait)  # Wait before retrying
            # Resubmit the function to the executor for a retry
            future = executor.submit(function, *args, **kwargs)
            try:
                return future.result(timeout=timeout_seconds)
            except concurrent.futures.TimeoutError:
                print("API call #2 timed out")
                raise

def process_transcriptions(df, approach, system_prompt, output_dir, output_file, model_name = 'gpt-4o', examples=None, test = False, num_questions_loser = 6, num_questions_winner = 7, **kwargs):
    """
    Process transcriptions using either a 0-shot or 1-shot approach.
    
    Parameters:
    - df: DataFrame containing the data to be processed.
    - approach: '0-shot' or '1-shot' to determine the processing method.
    - system_prompt: The system prompt used for both 0-shot and 1-shot.
    - output_dir: Directory to save the output files with blocks separated as <BLOCK N>.. </BLOCK N>.
        Suggested naming convention: 'splits_{approach}_{model_name}_{round}'.
    - output_file: Name of the output file to save the DataFrame with the processed data.
        Suggested naming convention: 'transcriptions_split_AI_{approach}_{model_name}_{round}.xlsx'.
    - model_name: Optional, specifies the model to be used in the function calls.
    - examples: dictionary of tuples of the form { 'winner': (examples), 'loser': (examples) }
    - test: set to True if for testing purposes, defaults to False
    - num_questions_loser: Optional, number of questions for the loser, defaults to 6 (good for both round 1 and 2)
    - num_questions_winner: Optional, number of questions for the winner, defaults to 7 for round 1, but should be set to 9 for round 2.
    
    **kwargs: Optional Keyword arguments:
    - temperature: float, defaults to 0.05

    Output:
    - DataFrame with the processed data saved to an Excel file with the specified name. The file is in wide format, with columns for each question and answer pair.
    - Output files with the blocks separated as <BLOCK N>.. </BLOCK N> saved to the specified directory. One file per row in the input DataFrame.
    """

    # Add tqdm to pandas
    tqdm.pandas()

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
        segments = construct_segments(row, num_losers=num_questions_loser, num_winners=num_questions_winner)
        if approach == '0-shot':
            try:
                output = call_api_with_timeout(split_questions_0shot, system_prompt, segments, row['full_transcription'], model=model_name, **kwargs)
            except Exception as e:
                print(f"Failed to process the API call for file {row['file_name']}:", str(e))
                continue
        elif approach == '1-shot':
            if examples is None:
                raise ValueError("examples argument is required if approach = '1-shot'")
            else:
                (u, a) = examples['winner'] if row['winner'] == 1 else examples['loser']
                try:
                    output = call_api_with_timeout(split_questions_1shot, system_prompt, segments, row['full_transcription'], u, a, model=model_name, **kwargs)
                except Exception as e:
                    print(f"Failed to process the API call for file {row['file_name']}:", str(e))
                    continue
        # File handling
        file_name = os.path.join(output_dir, f"split_{row['file_name'].replace(' ', '_')}.txt")
        with open(file_name, "w") as outfile:
            outfile.write(output)
        print("output saved to ", file_name)

        # Extract blocks and handle errors
        try:
            blocks, length = extract_blocks(output)
        except ValueError as e:
            logging.error(f"Error at index {index} with file {row['file_name']}: {str(e)}")
            df.at[index, 'error_message'] = str(e)
            df.at[index, 'correct_num_seg'] = False
            continue
        
        # Compute similarity (only if test is True) and update DataFrame
        df.at[index, 'avg_similarity'] = 0
        df.at[index, 'correct_num_seg'] = length == len(segments)
        for i, seg in enumerate(blocks):
            column_name = f'question_answer_{i+1}_{model_name}'
            df.at[index, column_name] = seg
            if length == len(segments) and test:
                try:
                    df.at[index, f'question_{i+1}_similarity'] = count_similarity(df.loc[index, [f'question_answer_{i+1}', f'question_answer_{i+1}_{model_name}']])
                    df.at[index, 'avg_similarity'] += df.at[index, f'question_{i+1}_similarity'] / length
                except ValueError as e:
                    print(f"Error at index {index} with file {row['file_name']}: {str(e)}")
                    print(f'Problematic question: {i+1}.\nText: "{df.loc[index, [f'question_answer_{i+1}']]}"')

    df.to_excel(output_file, index=False)
    print(f"Data saved to {output_file}")

def fill_question_details(row, question_map_loser, question_map_winner):
  """
  Fills question details (id and text) based on winner and question number.

  Args:
      row (pd.Series): A row from the DataFrame.
      question_map_loser (dict): Dictionary mapping question number to details for losers.
      question_map_winner (dict): Dictionary mapping question number to details for winners.

  Returns:
      pd.Series: A Series containing the filled question id and text.
  """

  question_num = row['question_num']
  winner = row['winner']

  if question_num == 3:
    question_id = row['randomizedquestionsmoney_do']
    question_text = {
        "financial_situation": "How does your financial situation now compare to a year ago?",
        "saving": "What is your personal experience with saving money?",
        "debts": "Do you have any debts? If yes, why did you take a loan?",
        "management_learn": "How do you learn about the best ways to manage your money?"
    }.get(question_id)
  elif question_num == 5:
    question_id = row['randomizedquestionshealth_do']
    question_text = {
        "happiness": "If you think about the past ten days, what are some things that make you happy? What are some things that make you sad?",
        "stress": "How does your current stress level compare to same time last year?",
        "alcohol": "Do you consume alcohol? If yes, how often, and why?"
    }.get(question_id)
  else:
    # Use appropriate question map based on winner
    question_map = question_map_winner if winner == 1 else question_map_loser
    question_id = question_map.get(question_num, {}).get("id")
    question_text = question_map.get(question_num, {}).get("text")

  return pd.Series({'question_id': question_id, 'question_text': question_text})



#####################################################################
####################### Embeddings and visualization ################
#####################################################################

### Embedding with openai models

def get_embedding(text, model = "text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

# Define a custom exception for timeout
class TimeoutException(Exception):
    pass

# Define a handler for the timeout
def timeout_handler(signum, frame):
    raise TimeoutException()

# Define a retry strategy
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type((Exception, TimeoutException)), reraise=True)
def get_embedding_with_retry(text, model = "text-embedding-3-small"):
    # Set a timeout of 10 seconds for each try
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(5)  # Set the alarm for 5 seconds
    try:
        result = get_embedding(text, model)
    finally:
        signal.alarm(0)  # Disable the alarm
    return result

def safe_get_embedding(x, row, column, model = "text-embedding-3-small"):
    if pd.isna(x):
        return None
    try:
        return get_embedding_with_retry(x, model=model)
    except Exception as e:
        print(f"Error processing row {row}, column {column}: {e}")
        return None

def safe_compute_embeddings(df, model = "text-embedding-3-small", columns_to_embed = ['transcription', 'ground_truth']):
    for c in columns_to_embed:
        c_emb = c +'_embedding'
        tqdm.pandas(desc=f'Embedding {c}')
        df[c_emb] = df.progress_apply(lambda row: safe_get_embedding(row[c], row.name, c, model = model), axis=1)
    return df

def compute_embeddings(df, model = "text-embedding-3-small", columns_to_embed = ['transcription', 'ground_truth']):
    for c in columns_to_embed:
        c_emb = c +'_embeddings'
        tqdm.pandas(desc=f'Embedding {c}')
        df[c_emb] = df[c].progress_apply(lambda x: get_embedding(x, model=model) if not pd.isna(x) else None)
    return df

### Embedding with voyager models

def get_embeddings_voyage(df, columns_to_embed = ["transcription"], batch_size= 32, input_type = "document"):
    """
    Embeds the specified columns of a DataFrame using the Voyage library.

    Args:
        df (pandas.DataFrame): The DataFrame containing the columns to embed.
        columns_to_embed (list, optional): The list of column names to embed. Defaults to ["transcription"].
        batch_size (int, optional): The batch size for embedding. Defaults to 32.
        input_type (str, optional): The type of input for embedding. Must be one of "document" (default), "query",
            "classification", or "clustering".

    Returns:
        pandas.DataFrame: The DataFrame with the embedded columns.

    Raises:
        ValueError: If the input_type is not one of the allowed values.

    """
    # raise an error if the input type is not valid
    if input_type not in ["document", "query", "classification", "clustering"]:
        raise ValueError("The input_type must be one of 'document', 'query', 'classification', or 'clustering'.")

    lists = dict( [(c, df[c].tolist()) for c in columns_to_embed] )
    if input_type == "classification":
        # prepend the string "Classify the text: " to each transcription
        lists = {k: [f"Classify the text: {v}" for v in lists[k]] for k in lists}
        input_type = None

    if input_type == "clustering":
        # prepend the string "Cluster the text: " to each transcription
        lists = {k: [f"Cluster the text: {v}" for v in lists[k]] for k in lists}
        input_type = None

    for c in columns_to_embed:
        print(f"Embedding column {c}. Total documents: {len(lists[c])}")
        embeddings = []
        for i in range(0, len(lists[c]), batch_size):
            batch = lists[c][i:i+batch_size]
            print(f"Embedding documents {i} to {i+batch_size}")
            print("Total tokens:", vo.count_tokens(batch, model = "voyage-large-2-instruct"))

            batch_embeddings = vo.embed(
                batch, model="voyage-large-2-instruct", input_type=input_type
            ).embeddings
            embeddings += batch_embeddings
        
        df[f"{c}_embedding"] = embeddings
    
    return df

### Visualization

def dim_reduction_tsne(df, metric = 'euclidean', columns_to_reduce= ['transcription_embeddings', 'ground_truth_embeddings'], n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200):
    tsne = TSNE(n_components=n_components, metric = metric, perplexity=perplexity, random_state=random_state, init=init, learning_rate=learning_rate)
    rs = {}
    for c in columns_to_reduce:
        matrix = np.array(df[c].to_list())
        c_name = "_".join(c.split("_")[:-1])
        rs[c_name] = tsne.fit_transform(matrix)
    return rs

def plot_tsne(df, color_variable, shape_variable=None, size_variable = None, plot_means = False, pca_dim = 50, metric = 'euclidean', **kwargs):
    """
    Plots t-SNE visualizations for given DataFrame with subplots for different models.

    Args:
    df (DataFrame): DataFrame containing the data.
    color_variable (str): Column name for the variable to use for coloring the points.
    shape_variable (str, optional): Column name for the variable to use for shaping the points.
    size_variable (str, optional): Column name for the variable to use for sizing the points.
    plot_means (bool, optional): Whether to plot the mean vectors for each group. Defaults to False.
    pca_dim (int, optional): Number of dimensions to reduce to using PCA decomposition. None if no pca decomposition is needed.
    metric (str): The metric to use when calculating the distance between instances in a feature array.
    tsne_kwargs (dict): Keyword arguments for the t-SNE dimensionality reduction function.
    columns_to_reduce (list or None): Optional list of columns to use for dimensionality reduction.

    Returns:
    None, displays a matplotlib plot.
    """

    # check if kwargs has columns_to_reduce
    if 'columns_to_reduce' in kwargs:
        columns_to_reduce = kwargs['columns_to_reduce']
    else:
        columns_to_reduce = ['transcription_embeddings', 'ground_truth_embeddings']

    if pca_dim is not None:
        # Create a new dataframe to store the modified columns
        new_df = df.copy()
        
        # First reduce to pca_dim using PCA decomposition
        pca = PCA(n_components=pca_dim)
        for c in columns_to_reduce:
            matrix = np.array(df[c].to_list())
            # compute PCA of column c and assign to new_df[c]
            new_df[c] = pca.fit_transform(matrix).tolist()
        
        # Dimensionality reduction, passing new_df to dim_reduction_tsne
        vis_dims = dim_reduction_tsne(new_df, metric=metric, **kwargs)
    else:
        # t-sne dimensionality reduction using cosine similarity
        vis_dims = dim_reduction_tsne(df, metric= metric, **kwargs)    

    # Prepare the subplot grid
    num_models = len(vis_dims)
    nrows = int(np.ceil(num_models / 2))  # arrange plots in 2 columns
    ncols = 2 if num_models > 1 else 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 5 * nrows))
    if ncols == 1:
        axes = [axes]  # convert to a list for consistent iteration
    else: 
        axes = axes.flatten()  # Flatten the array of axes if necessary

    # Loop through the dictionary to create a plot for each model
    for idx, (model_name, coordinates) in enumerate(vis_dims.items()):
        # Creating an embedded DataFrame with necessary variables for plotting
        columns_needed = [color_variable]
        if shape_variable:
            columns_needed.append(shape_variable)
        if size_variable:
            columns_needed.append(size_variable)

        df_toplot = pd.DataFrame({"x": coordinates[:, 0], "y": coordinates[:, 1]})
        df_toplot = pd.concat([df_toplot, df[columns_needed].reset_index()], axis=1)
        
        # Create t-SNE plot on the current subplot
        g = sns.scatterplot(
            x="x",
            y="y",
            hue=color_variable,
            style=shape_variable if shape_variable else None,
            size=size_variable if size_variable else None,
            #sizes=(20, 200),  # Adjust sizes range as needed
            data=df_toplot,
            palette="Set1",
            alpha=0.7,
            ax=axes[idx],
        )
        
        # Plot mean vectors
        if plot_means:
            # mean_vectors = df_toplot.groupby([color_variable, shape_variable])[['x', 'y']].mean().reset_index() if shape_variable else df_toplot.groupby(color_variable)[['x', 'y']].mean().reset_index()
            mean_vectors = df_toplot.groupby(color_variable)[['x', 'y']].mean().reset_index()
            sns.scatterplot(
                x="x",
                y="y",
                hue=color_variable,
                # style=shape_variable if shape_variable else None,
                data=mean_vectors,
                palette="winter",
                marker='X',
                s=200,
                ax=axes[idx],
            )

        # Customize plot title and labels for each subplot
        g.set_title(f"t-SNE for {model_name}")
        g.set_xlabel("t-SNE Dimension 1")
        g.set_ylabel("t-SNE Dimension 2")

    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Place legend at the 'upper right' corner outside the plot box
    plt.legend(bbox_to_anchor=(1.02, 1))  # You can adjust the values to move the legend around
    plt.show()



#####################################################################
######################### API Call testing ##########################
#####################################################################

def main():
    # example API call: ask gpt-3.5 to answer a call. Just to test the API connection
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI assistant. Answer 'API call successful' if you receive this message."},
            {"role": "user", "content": "This is just a test API call. Please answer if the call was successful."}
        ]
    )
    print(response.choices[0].message.content)

if __name__ == "__main__":
    main()
