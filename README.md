# can_ai_learn_basic_functions
WARNING: RESEARCH REPO

We ablate many different proposed methods just to learn basic functions like addition, subtraction, multiplication, division, copy, concatenation, etc, without COT (Chain of Thought) or calculator tokens. 

Surprisingly, standard SGD on LLMs can not solve! The only things that have worked are 1) Char Tokenizer or Word Tokenizer with spaced input + LLMs like pythia + SGD. 2) NTMs/DNCs and char tokenizer. MeZO dramatically outperforms on these tasks with almost no VRAM!! Crazy. How have ppl not done this before??? Writing ICML paper now..

To run, you should use the sweep shell scripts. Ensure you first _chmod +x *.sh_. 

Then you can run with _./*.sh_
