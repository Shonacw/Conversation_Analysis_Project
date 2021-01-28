# Master's Project: Conversation Structure Analysis + Visualisation using Podcast Transcripts

In this MSci project we are looking at the structure of human-human conversation under 2 different lenses: as a macroscopic trajectory through a topic space (built from the components of word embeddings) and as a string of microscopic dialogue acts. From the topic space analysis we hope to build a signature graphical representation of the transcript which visualises the evolution of topics discussed, and portrays key information about the given conversation. From the dialogue analysis we hope to answer the question 'What makes Conversation Interesting?'.

I am focussing on the Topic-related tasks of this project (Segmentation, Embeddings, and Visualisation). Code for the Dialogue-Act analysis part of this project can be found on Jonas Scholz' Github: https://github.com/jonas-scholz123/msci-project 

*Key libraries used: sklearn, torch, gensim, nltk, spacy, scipy, pandas, networkx*

# Topic Segmentation and Embeddings
Steps taken so far...

*1) Keyword Extraction*
- Keywords and phrases, using PKE implementation of TopicRank.
- Nouns, using spacy POS tagger with the en_core_web_sm pretrained statistical model. 
- Bigrams and Trigrams, using NLTK implementation of Collocation Finder.

*2) Topic Space Construction*
- Word2Vec implementation with GoogleNews-vectors-negative300 pretrained word embeddings.
- FastText implementation with the cc.en.300.bin pretrained model.

*3) Transcript Segmentation*
- Infersent implementation with arbitrary cosine similarity cutoff between the embeddings of consecutive sentences.
- SliceCast implementation.
- Even chunks, option for statistical analysis.

*4) Graphical Representation*
- Labelled word embeddings of keywords extracted from the given transcript 
- Quiver plot of trajectory taken by conversation through topic space 
- 3D Quiver plot of trajectory taken by conversation through topic space with Sentence Number on the z axis representing time.


## Useful Links
Embedding techniques used:
* Sentence embeddings with [InferSent](https://github.com/facebookresearch/InferSent) developed by Facebook Research for utterance-level analysis, from the paper [Supervised Learning of Universal Sentence Representations from Natural Language Inference Data](https://arxiv.org/abs/1705.02364). 
* Word embeddings with [Word2Vec](https://arxiv.org/abs/1301.3781) for EDU-level analysis, uing [this](https://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/) Word2Vec model pretrianed by Google.
* Word embeddings with [FastText](https://github.com/facebookresearch/fastText) based on the paper [Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606)for EDU-level analysis, using the [cc.en.300.bin](https://fasttext.cc/docs/en/crawl-vectors.html) model.

Segmentation methods used: 
* [SliceCast](https://github.com/bmmidei/SliceCast) implementation from [Neural Text Segmentation on Podcast Transcripts](https://github.com/bmmidei/SliceCast/blob/master/Neural_Text_Segmentation_on_Podcast_Transcripts.pdf).
* [Infersent](https://github.com/facebookresearch/InferSent) sentence embeddings paired with a variable cosine similarity cutoff. Graphical method inspired by paper [Minimum_Cut_Model_for_Spoken_Lecture_Segmentation](https://www.researchgate.net/publication/220873934_Minimum_Cut_Model_for_Spoken_Lecture_Segmentation).


# Discussion Trees: Visualising Conversation Structure and Topic Evolution
This part of the project investigates how one can best visualise the evolution of ideas and nature/flow of conversation from a given podcast transcript. The output graphic will act as a visual snapshot of the conversation, providing a viewer with insight into the extent to which different topics were discussed. The goal is to make key themes and points discussed immediately accessible, highlighting which discussion points need to be built out and which have yet to be explored all at just a glance.

Questions to answer:
- Which key topics were revisited during the conversation?
- What was the variation in Dialog Act usage over the course of the conversation (segment-wise), and what does this tell us about the nature/ flow of the conversation? 
- How do we quantify the efficiency of a conversation? (Mostly relevant when analysing business meeting transcripts)
- From the conversation visualisations can we detect a set of common trajectories taken by conversations of similar styles?


## Word Embedding-based Visualisation methods attempted so far...
***Please note that this is a work in progress!***

How the location of a node representing a segment of the transcript is assigned for plots modelling topic evolution:
- Average of all the word-vectors of the keywords present in the given segment
- The word-vector of only the most frequent keyword from the segment
- Average of word-vectors corresponding to the three most frequently ocurring keywords in the given segment

How the thickness of the line is assigned (todo):
- Dialog acts? Representation of speaker uncertainty? 


**Plotting the Word Embeddings**

   Here the keywords are extracted from the given transcript using methods listed in 1) and are plotted
   in word-embedding space using the cc.en.300 FastText pretrained model.
  
  <img src = "Saved_Images/Fasttext_WordEmbedding_ShiftedNgrams_wide.png" width ="700" /> 
  And a zoomed-in version...
  <img src = "Saved_Images/FastText_WordEmbedding_ZOOMED.png" width ="700" />
  And another...
  <img src = "Saved_Images/FastText_WordEmbedding_ZOOMED_2.png" width ="700" />
   
**Plotting of Trajectory Through Topic Space**

   The following example was created by segmenting the transcript into (left image) 20 and (right image) 100 even sections, calculating 
   the position of the nodes using the average of the top 3 keywords used in each section...
    
   <img src = "Saved_Images/20_Even_Segments_Quiver_Plot_With_3_max_count_NodePosition.png" width ="400" /> <img src = "Saved_Images/100_Even_Segments_Quiver_Plot_With_3_max_count_NodePosition.png" width ="400" />

   The same trajectory plotted in Word2Vec word-embedding space with key nouns labelled. Here we have (left image) 20- and (right image) 100-even segments using average of 3 top keyword word-vectors as the node position for each segment. 
    
   <img src = "Saved_Images/20_Even_Segments_Quiver_and_Embeddings_Plot_With_3_max_count_NodePosition.png" width ="400" /> <img src = "Saved_Images/100_Even_Segments_Quiver_and_Embeddings_Plot_With_3_max_count_NodePosition.png" width ="400" />
    
    
**Plotting of Trajectory Through 3D Topic Space**

20_Even_Segments_3D_Quiver_With_3_max_count_NodePosition 

<img src = "Saved_Images/20_Even_Segments_3D_Quiver_With_3_max_count_NodePosition.png" width ="400" />

**Plotting of Trajectory Through Topic Space for two Transcripts on the same background word embedding**

Word embedding background created using FastText on the two transcript (Joe Rogan + Elon Musk, and Joe Rogan + Jack Dorsey).
<img src = "Saved_Images/combined_podcast/joe_rogan_elon_musk/200_Even_Segments_Quiver_and_Embeddings_Plot_With_3_max_count_NodePosition.png" width ="400" /> <img src = "Saved_Images/combined_podcast/joe_rogan_jack_dorsey/200_Even_Segments_Quiver_and_Embeddings_Plot_With_3_max_count_NodePosition.png" width ="400" />


**Conversation Trajectory with 'Richness' of Segment**

Added variable sizing for the text labels of Nodes with multiple connections, where the size of text reflects the number 
of connections the node has / the size of the cluster. Words in bold correspond to node locations visited more than once.
Added colour to the lines to reflect the number of keywords contained in each segment. More keywords contained means 
'richer' conversation in this section, but also means there is a larger uncertainty on the topic of this segment in that 
there were more keywords which could have been chosen to define the segment node position. Both plotted on their shared
 word embedding space. 

Corresponding histograms for the number of keywords contained in each segment of the transcripts ... 
a) Joe Rogan & Elon Musk                                         b) Joe Rogan & Jack Dorsey

<img src = "Saved_Images/combined_podcast/joe_rogan_elon_musk/histogram_of_keywords.png" width ="400" /> <img src = "Saved_Images/combined_podcast/joe_rogan_jack_dorsey/histogram_of_keywords.png" width ="400" />

Elon Musk Trajectory...

<img src = "Saved_Images/combined_podcast/Elon_Musk_100.png" width ="900" />

Jack Dorsey Trajectory...

<img src = "Saved_Images/combined_podcast/Jack_Dorsey_100.png" width ="900" />

**Plotting of Speaker-Indepedent Trajectory Through Topic Space**

Separated the utterances spoken by each member of the conversation and plotted their independent topical trajectories. Example
given here is for the Joe Rogan & Elon Musk interview.

<img src = "Screenshots/Screenshot 2021-01-21 at 09.57.23.png" width ="900" />


## A Second method of Visualisation: Back to Basics
**Dialogue Act Simple Line**

Testing basic hypothesis that Questions indicate a topic change in the conversation. Here, I plot a node every time the speaker changes.
 I check through all dialogue acts used by a speaker in their set of consecutive utterances and if they used a Question 
 ("Wh-Question", "Yes-No-Question", "Declarative Yes-No-Question", "Declarative Wh-Question")
I shift the trajectory one step in the y direction and annotate the node with the Topic currently being discussed. If no
questions were asked then I assume the overall topic being discussed has not changed and continue plotting the nodes
along the horizontal axis. Here only the first 200 utterances in the transript are considered. I also
colour the lines to reflect the current speaker -> from this we can see that all of the Questions asked in the first
200 lines of the transcript are spoken by Joe Rogan (and hence maybe he introduces all new topics as he is leading
the conversation?).

<img src = "Saved_Images/joe_rogan_elon_musk/Simple_Line_DA.png" width ="700" />

**Topic Changes Simple Line**

Here I instead shift the line up when we detect the topic of conversation has changed. We can see that although 
**Elon Musk** does not ask many questions, he DOES lead the conversation in terms of its topical evolution (all the blue lines
 on the steps). Data presented in table format below on the number of Topics introduced by each speaker in the conversetion
 as a whole.

<img src = "Saved_Images/joe_rogan_elon_musk/Simple_Line_Topics.png" width ="700" />

| Speaker   |   Total #Utterances |   Total #Questions Asked |   Total #Statements |   Number of Topics Introduced |   #Topics introduced by Statement |   #Topics introduced by Question |
|:----------|--------------------:|-------------------------:|--------------------:|------------------------------:|----------------------------------:|---------------------------------:|
| Joe Rogan |                 854 |                      161 |                 519 |                            74 |                                50 |                               20 |
| Elon Musk |                 847 |                       29 |                 591 |                            80 |                                64 |                                3 |


and same for **Jack Dorsey** interview...

<img src = "Saved_Images/joe_rogan_jack_dorsey/Simple_Line_Topics.png" width ="700" />

| Speaker     |   Total #Utterances |   Total #Questions Asked |   Total #Statements |   Number of Topics Introduced |   #Topics introduced by Statement |   #Topics introduced by Question |
|:------------|--------------------:|-------------------------:|--------------------:|------------------------------:|----------------------------------:|---------------------------------:|
| Joe Rogan   |                 727 |                      158 |                 387 |                            63 |                                32 |                               23 |
| Jack Dorsey |                 992 |                       47 |                 775 |                            82 |                                70 |                                1 |



**Topic Changes Line with Shift**

Take the simple line plot above and introduce a horizontal shift when a previous topic is detected. 

<img src = "Saved_Images/joe_rogan_elon_musk/Shifting_Line_Topics.png" width ="600" />

...and a simplified version...

<img src = "Saved_Images/joe_rogan_elon_musk/Simpler_Shifting_Line_Topics.png" width ="600" />

and now same for the Jack Dorsey interview

<img src = "Saved_Images/joe_rogan_jack_dorsey/Shifting_Line_Topics.png" width ="400" /> <img src = "Saved_Images/joe_rogan_jack_dorsey/Simpler_Shifting_Line_Topics.png" width ="400" />

*Topic Changes Line with Shift - Comparison between different interviews*

And now similar but for the ENTIRE conversations between a) Joe Rogan and Jack Dorsey, b) Joe Rogan and Elon Musk

<img src = "Saved_Images/joe_rogan_elon_musk/DT_Shifting_Topic_Line_LogX:False.png" width ="400" /> <img src = "Saved_Images/joe_rogan_jack_dorsey/DT_Shifting_Topic_Line_LogX:False.png" width ="400" />

Now introducing a logarithmic x-axis scale so can view the dense information on the LHS of the above graphs better....

<img src = "Saved_Images/joe_rogan_elon_musk/DT_Shifting_Topic_Line_LogX:True.png" width ="400" /> <img src = "Saved_Images/joe_rogan_jack_dorsey/DT_Shifting_Topic_Line_LogX:True.png" width ="400" />

Unreadable graph, but noticing the difference in pattern between that of above two conversations, 
and the interview between Joe Rogan and Kanye West...

<img src = "Screenshots/Screenshot 2021-01-24 at 17.57.22.png" width ="500" />



**Discussion Trees - Rearranging the Simple Shift Visualisation**

Rather than *shifting* to an old topics position when it is returned to, we start a new branch from 
its last position... (branch numbers annotated in the orange circles)

<img src = "Screenshots/Screenshot 2021-01-28 at 14.11.42.png" width ="400" />

Each time we start a new branch, send the x increments a different way along the x axis....

<img src = "Screenshots/Screenshot 2021-01-28 at 14.10.49.png" width ="400" />

Angle the horizonal components of each branch with an incrementally steeper gradient such 
that we can differentiate different branches....

<img src = "Screenshots/Screenshot 2021-01-28 at 14.10.49.png" width ="400" />

Also add a slight colour map to help differentiate them...

<img src = "Screenshots/Screenshot 2021-01-28 at 14.27.41.png" width ="400" />

And now shift down...




# Other Stuff
- **Interruptions... when one speaker cuts the other off**

<img src = "Saved_Images/joe_rogan_elon_musk/Interruptions_Elon.png" width ="350" /> <img src = "Saved_Images/joe_rogan_jack_dorsey/Interruptions_Jack.png" width ="350" />

<table>
<tr><th>Interruptions Elon Interview</th><th>Interruptions Jack Interview</th></tr>
<tr><td>

| Speaker   |   Total #Interruptions |
|:----------|-----------------------:|
| Joe Rogan |                      4 |
| Elon Musk |                      4 |

</td><td>

| Speaker     |   Total #Interruptions |
|:------------|-----------------------:|
| Joe Rogan   |                     53 |
| Jack Dorsey |                     58 |

</td></tr> </table>

Does this tell us something about the nature of the conversation? Help us classify the conversation? insight into how 
comfortable the interviewer feels around the interviewee?

- **Snappyness of conversation (other word needed): basic illustration using utterance lengths**

<img src = "Saved_Images/Stuff/elon_musk_n:100_normalised:False.png" width ="350" /> <img src = "Saved_Images/Stuff/jack_dorsey_n:100_normalised:False.png" width ="350" />

Tells us something about the pace of human conversations - is it common for the pace to slow in the 2nd quarter? for it 
to speed up just before winding down at the end?... how can we quantify this and use it as a feature for conversation classification?

- **Learning how to create Quiver animations...**

(note VERY MESSY- not final versions in any way)

<img src = "Animations/myAnimation.gif" width ="600" />

<img src = "Animations/myAnimation_4S.gif" width ="600" />

^speaker-wise trajectories 


# Code Usage 

Run \_\_main__.py with the following params

- *path_to_transcript*: Local path to the transcript you wish to investigate.
- *embedding_method*: Word Embedding of choice. Options are 'fasttext' or 'word2vec'. 
- *seg_method*: Method of segmentation to use. Options are 'Even' (for splitting it evenly into n segments), 'InferSent' for using InferSent with 
    a certain cosine similarity cut-off, or 'SliceCast'.
- *node_location_method*: How to define the node position for a given segment... options are 'total_average' (i.e. calculate
the average of the keyword-vectors present in each segment), '1_max_count' for choosing just the most-frequent keyword's embedding
as the segment node location, or '3_max_count' for taking the word vector average of the three most frequency used keywords 
in each segment as the node location.

- *Even_number_of_segments*: The number of even segments for the transcript to be split into, use when seg_method = 'Even'.
- *InferSent_cos_sim_limit*: The value of cosine similarity at which two consecutive sentences which be deemed as not-similar-enough
to belong to the same section (and hence a new-section tag will be placed between them). Use when for when seg_method = 'InferSent'.

- *Plotting_Segmentation*: True or False, whether you want to plot visualisation of the segmentation task.
- *saving_figs*: True or False, whether you want to save figures produced on this  run.
