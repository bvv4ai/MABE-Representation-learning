MABE challenge - Social action recognition in mice
Final demo video
out.mp4

=> build models to identify over 30 different social and non-social behaviors in pairs and groups of co-housed mice. 

The data we got is from many labs, we need to identify an effective common way to process them.
We have atmost 4 mice in each setup. We have the xy locations of the mouse bodypart.

My approach is to recreate the top down view of arena.

Based on shapes, arena can be classified into rectangular,sqaure, circular.

Based on familiarity of the arena to the mice, it is classified as following:
types_of_arenas = {"neutral": 153/255.0
                   ,"resident-intruder":77/255.0
                   ,"divided territories":97/255.0,
                   "familiar": 34/255.0,
                   "CSDS":67/255.0,np.nan:128/255.0}
My approach is to first create the arena, and color the arena (used grayscale) based on its type.

Mice can be classified into 2 types (Male,female) based on gender. To distinguish the gender,
the male mice I draw on the arena will have large black ears.
Based on color, the mice can further be classified as
color_dict = {'white':162/255.0,
              'black':183/255.0,
               'brown':222/255.0,
               'black and tan':15/255.0,
               np.nan:	175/255.0}
Each color mouse is represented with a gray scale value when we draw on our canvas
Further based on genus, mice are classified belonging to strains
strain_dict = {'CD-1 (ICR)':212/255.0,
                'C57Bl/6N' : 120/255.0,
               'C57Bl/6J':  190/255.0, 
               '129/SvEvTac':	235/255.0,
                 'C57Bl/6J x Ai148':197/255.0,
                 'BTBR': 200/255.0,
                   'CD1':148/255.0,
                     'CFW': 110/255.0,
                     'BALB/c':46/255.0,}
To distinguish the different strains, we draw a strain line from a mouse leaft ear to its right ear.
