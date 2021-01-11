## Master's Project: Conversation Structure Analysis + Visualisation using Podcast Transcripts

In this MSci project The goal is to extract information about conversations by plotting the 
investigating topic evolution 
what key topics were revisited during the conversation? 
are there a set of common trajectories similar- conversations will take

Specifically, we are looking at the structure of conversation under 2 different lenses: as a macroscopic trajectory through a topic space (built from the components of word embeddings) and as a string of microscopic dialogue acts.

Code for the Dialogue Analysis part of this project can be found here: https://github.com/jonas-scholz123/msci-project (Jonas Scholz' Github)

## Topic Segmentation and Embeddings
Steps taken so far...

*1) Keyword Extraction*

*2) Topic Space Construction*

*3) Transcript Segmentation*

*4) Preliminary Topic Exploration: Plotting*

Embedding techniques used:
* Sentence embeddings with InferSent (Facebook Research, https://github.com/facebookresearch/InferSent) for utterance-level analysis. 
* Word embeddings with Word2Vec for EDU-level analysis.

Segmentation methods used: 
* Infersent 
* SliceCast
* Even Segments 




## Discussion Trees: Visualising Conversation Structure and Topic Evolution
This part of the project investigates how we can best visualise the evolution of ideas and nature/flow of conversation from a given podcast transcript. The output graphic will represent a visual snapshot of the conversation, providing a viewer immediate insight to the extent to which different topics were discussed. It will make key themes and points discussed accessible, highlighting which points need to be built out and which have yet to be explored all at just a glance. 

