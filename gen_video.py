import numpy as np
import pandas as pd
import cv2
import itertools
import ast
import time
import math
import sys
#sys.path.append('../models/')
#from feat_extract import SqueezeNetFeatureExtractor
import torch

strain_dict = {'CD-1 (ICR)':212/255.0,
                'C57Bl/6N' : 120/255.0,
               'C57Bl/6J':  190/255.0, 
               '129/SvEvTac':	235/255.0,
                 'C57Bl/6J x Ai148':197/255.0,
                 'BTBR': 200/255.0,
                   'CD1':148/255.0,
                     'CFW': 110/255.0,
                     'BALB/c':46/255.0}
color_dict = {'white':162/255.0,
              'black':183/255.0,
               'brown':222/255.0,
               'black and tan':15/255.0,
               np.nan:	175/255.0}
sex_dict = {'male':1, 'female':0 }
types_of_arenas = {"neutral": 153/255.0
                   ,"resident-intruder":77/255.0
                   ,"divided territories":97/255.0,
                   "familiar": 34/255.0,
                   "CSDS":67/255.0,np.nan:128/255.0}

male_color = 	0
mouse_id_color = {1:51/255.0,
                  2:8/255.0,
                  3:25/255.0,
                  4:86/255.0}
classes = ['bkg','rest', 'disengage', 'reciprocalsniff', 'attack', 'defend',
            'sniffbody', 'flinch', 'dominance', 'ejaculate', 'avoid', 'submit', 
            'mount', 'freeze', 'chase', 'intromit', 'huddle', 'rear', 'genitalgroom', 
            'run', 'dominancegroom', 'exploreobject', 'dominancemount', 'sniff', 'sniffgenital', 
            'follow', 'escape', 'approach', 'sniffface', 'climb', 'shepherd', 'chaseattack',
              'allogroom', 'attemptmount', 'biteobject', 'selfgroom', 'dig', 'tussle']
class_mapping = {c: i for i, c in enumerate(classes)}
num_classes = len(classes)
num_mouses = 4

class make_mouse:
    def __init__(self,params_dict,mouse_idx):
        self.mouse_idx = mouse_idx
        self.mouse_idx_color = mouse_id_color[mouse_idx]
        self.mouse_color = color_dict[params_dict['mouse_color']]
        self.mouse_strain = strain_dict[params_dict['mouse_strain']]
        self.mouse_sex = sex_dict[params_dict['mouse_sex']]
        self.body_parts_tracked = params_dict['body_parts_tracked']
        #self.body_combs = list(itertools.combinations(self.body_parts_tracked,2))
        
    def draw_mouse_on_canvas(self,canvas,position_dict):
        for body_part in self.body_parts_tracked:
            pt1 =  position_dict[body_part]
            
            if pt1 == (-1,-1) :
                continue    
            if  self.mouse_sex:
                if (body_part == 'ear_left' or body_part == 'ear_right'):
                    cv2.circle(canvas, pt1, 7, male_color, -1)
                else:   
                    cv2.circle(canvas, pt1, 3, self.mouse_color, -1)    
            else:

                cv2.circle(canvas, pt1, 3, self.mouse_color, -1)
                
        cv2.line(canvas, position_dict['ear_left'], position_dict['ear_right'], self.mouse_strain, 6)
        #cv2.rectangle(canvas, position_dict['min_xy'], position_dict['max_xy'], self.mouse_idx_color, 2)
        
                
        return canvas

class video_maker:
    def __init__(self,tracking_file,video_params,annot_file=None,resize_canvas=None):
        self.track_df = pd.read_parquet(tracking_file)
        
        self.arena_type = types_of_arenas[video_params['arena_type'].item() ]     
        self.arena_shape = video_params['arena_shape'].item()
        self.fps = video_params['frames_per_second'].item()
        if resize_canvas:
            self.scale_width = resize_canvas[1]/video_params["video_width_pix"].item()
            self.scale_height = resize_canvas[0]/video_params["video_height_pix"].item()
        else:
            self.scale_width = 1
            self.scale_height = 1
        self.video_width = int(video_params["video_width_pix"].item()*self.scale_width)
        self.video_height = int(video_params["video_height_pix"].item()*self.scale_height)
        self.arena_width = int(video_params["arena_width_cm"].item()* video_params["pix_per_cm_approx"].item())
        self.arena_height = int(video_params["arena_height_cm"].item() * video_params["pix_per_cm_approx"].item() )
        self.video_dur = video_params['video_duration_sec'].item()
        self.mouse_present = [not pd.isna(video_params['mouse1_strain'].item()),
                              not pd.isna(video_params['mouse2_strain'].item()),     
                                not pd.isna(video_params['mouse3_strain'].item()),
                                    not pd.isna(video_params['mouse4_strain'].item())]
        if annot_file is not None:
            self.annot = pd.read_parquet(annot_file)
        else:
            self.annot = None
        #print(self.annot.head())
        #exit()
        
        self.mouse_obj = []
        body_parts_tracked = ast.literal_eval(video_params['body_parts_tracked'].item())
        for index_mouse in range(len(self.mouse_present)):
            if self.mouse_present[index_mouse]:
                mouse_dict = {'mouse_color':video_params['mouse'+str(index_mouse+1)+'_color'].item(),
                      'mouse_strain':video_params['mouse'+str(index_mouse+1)+'_strain'].item(),
                      'mouse_sex':video_params['mouse'+str(index_mouse+1)+'_sex'].item(),
                      'body_parts_tracked':body_parts_tracked}
                self.mouse_obj.append(make_mouse(mouse_dict,index_mouse+1))
                
                
            


    def create_arena_canvas(self):
        self.canvas = np.zeros((self.video_height,self.video_width,1), np.float32)
        k = (self.video_width/self.scale_width - self.arena_width)//2
        m = (self.video_height/self.scale_height - self.arena_height)//2
        thickness = -1
        color = self.arena_type   
        if (self.arena_shape == "rectangular") or (self.arena_shape == "split rectangluar")or (self.arena_shape == "square"):
            start_point = (m,k)  
            end_point = (m+ self.arena_height, k+ self.arena_width)           
            start_point = self.change_point_scale(start_point)
            end_point = self.change_point_scale(end_point)
            cv2.rectangle(self.canvas, start_point, end_point, color, thickness) 
        else:
          radius = int(((self.scale_width+self.scale_height)//2*self.arena_width)//2)
          center = (m+radius,k+radius)
          center = self.change_point_scale(center)
          cv2.circle(self.canvas, center, radius, color, thickness)
    def change_point_scale(self,point):
        x = int(point[1]*self.scale_width)
        y = int(point[0]*self.scale_height)
        return (x,y)

    def get_xy(self,body_part,df_frame):
        df_frame = df_frame[df_frame['bodypart']==body_part]
        if(len(df_frame)==0):
            return (-1,-1)
        x = int(df_frame['x'].item()*self.scale_width)
        y = int(df_frame['y'].item()*self.scale_height)
        return (x,y)

        
    def start_vid(self,gen_video=False):
        self.create_arena_canvas()
        data_packet = {}
        t = (np.arange(0, self.video_dur,1)*self.fps).astype(int)
        data = []   
        if gen_video:
            codec = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 10
            video_writer = cv2.VideoWriter('./out.mp4', codec, fps,(self.video_width,self.video_height),isColor= False)
        for vid_frame_id in t:
            
            df_frame = self.track_df[self.track_df['video_frame']==vid_frame_id]
            frame_canvas = self.canvas.copy()
            for mouse in self.mouse_obj:
                position_dict = {}
                df_frame_mouse = df_frame[df_frame['mouse_id']==int(mouse.mouse_idx)]
                position_dict['min_xy'] = (int(df_frame_mouse['x'].min()*self.scale_width), int(df_frame_mouse['y'].min()*self.scale_height))
                position_dict['max_xy'] = (int(df_frame_mouse['x'].max()*self.scale_width), int(df_frame_mouse['y'].max()*self.scale_height))
        
                for body_part in mouse.body_parts_tracked:
                    
                    pos = self.get_xy(body_part,df_frame_mouse)
                    position_dict[body_part] = pos
                frame_canvas = mouse.draw_mouse_on_canvas(frame_canvas,position_dict)
            #frame_canvas = torch.from_numpy(frame_canvas).squeeze().unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
            
            #with torch.no_grad():
                #features = model(frame_canvas.float())  
                #data.append(features.squeeze(0))  
            
            new_frame = (255*frame_canvas).astype(np.uint8)
         
            #cv2.imshow('frame',new_frame)
            #cv2.waitKey(0)
            if gen_video:
                video_writer.write(new_frame)
        #data = torch.stack(data, dim=0)  
        
        #return data
            #cv2.imshow('frame',frame_canvas)
            #cv2.waitKey(0)
        video_writer.release()
        return 0

            
#model = SqueezeNetFeatureExtractor()
#model.eval() # Set to evaluation mode
lab_id = "UppityFerret"
file_name = "./1085105007Uppity.parquet"
video_id = 1085105007


df_params = pd.read_csv('./train.csv')
df_params = df_params[df_params['lab_id']==lab_id]
df_params = df_params[df_params['video_id']==video_id]
annot = "./1085105007.parquet"
start = time.time()
gener_video = video_maker(file_name,df_params)
#end = time.time()
#print("time taken ",end-start)
#exit().
#start = time.time()
data = gener_video.start_vid(True)
end = time.time()
print("time taken ",end-start)





































































































































































































































































































































































































        
