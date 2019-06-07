# fastai_2019_pt2_dev_course

Refactored Fast.ai Pt2 2019 with exported .py files all in a centralised location, exported all notebooks as scripts so that source code goto definition can be used in an IDE.
 
(I found it very difficult to follow the .ipynb's with chained exports and without being able to step to definition when required).

<pre>
Workflow:
    Work through notebooks
    If make any changes: 
        Export as .py to dl2/scripts/
        Run dl2/scripts/clean_nb_output.py to remove irrelevant ipynb code
    Step to relevant source in .py file using keybaord shortcuts
    Optionally debug .py file with breakpoint if difficult to follow
    NB debug only works pre .to_device, once on GPU not on CPU to be able to set breakpoints
</pre>

<pre>
dl2/
    .ipynb's with minor changes (added comments and fixes to improve readability for me)
    /utils
        Python files containing functions and classes all in one place
        Ideally would organise these files better but hacked it to get get through in reasonable time
    /scripts
        exported .ipynb's as .py to use in ide for debugging/stepping to source in ide 
    /exp
        at lesson 12 stage I found it too hard to keep traing changes so just exported all .py scripts in exp into one file, 
        ran black code formatter on it and imported that file (nb_formatted.py).
</pre>