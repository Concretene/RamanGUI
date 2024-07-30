from tkinter import Tk, Label, PhotoImage, Button, Canvas, Toplevel, Entry, messagebox, StringVar, OptionMenu, END, Text
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter
from scipy.signal import find_peaks, peak_widths, peak_prominences
import seaborn as sns
import matplotlib.transforms as tr
import re
from PIL import Image
from PIL import ImageTk

def labels_entry_canvas(window, canvas, label_name, x, y):
    """Create a label linked to a user entry window

    Parameters
    ----------
    window : tkinter.Tk
        
    canvas : tkinter.Canvas
        
    label_name : str
        
    x : int
    x coordinate
        
    y : int
    y coordinate
        

    Returns
    -------
    The label and a entry user window
    """


    label_generic = Label(window, text=label_name, font=("Segoe UI Variable Text Semiligh", 12))
    entry_generic = Entry(window)
    return entry_generic, canvas.create_window(x, y, window=label_generic), canvas.create_window(x + 150, y, window=entry_generic)

def open_preprocess_file(path):
    """Open a file and change the data to fit in a template

    Parameters
    ----------
    path : str
        

    Returns
    -------

    """
    print(f"Preprocess file open with {path}")
    with open(path) as file:
        lines = file.readlines()
        file_head = lines[0]
        replace_file_head = file_head.replace("\t\t","\t")
        new_file = [row for row in lines]
        new_file[0] = replace_file_head
        return(new_file)
    
def saving_preprocess_file(path, new_file):
    """Save the data after been process
    Parameters
    ----------
    path : str

    new_file : str

    Returns
    -------
    """
    print(f"Saving process file")
    rsplit_path = path.rsplit(sep='/',maxsplit=1)
    print(rsplit_path)
    relative_path = rsplit_path[0] + '/process_file/'
    file = rsplit_path[1].split(sep='.')
    file_name = file[0] + "_process." + file[1]
    file_process_path = relative_path + "/" + file_name
    file_process = open(file_process_path,'w')
    file_process.writelines(new_file)
    file_process.close
    print(f'File {file[0]} saved!')
    return file_process_path

def get_headers(path):
    """Processing the headers of the file

    Parameters
    ----------
    path : str
        

    Returns
    -------

    """
    with open(path) as file:
        lines = file.readlines()
        file_head = lines[0]
        strip_file_head = file_head.strip()
        #print(strip_file_head)
        replace_file_head = strip_file_head.replace("#","")
        #print(replace_file_head)
        replace_file_head = replace_file_head.replace("\t",",")
        replace_file_head = replace_file_head.replace(",,",",")
        split_file_head = replace_file_head.split(sep=",")
        #print(split_file_head)
        return split_file_head

def baseline_als(y, lam, p, niter=100):
    '''Baseline estimation algorithm based on asymetric least squares '''
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

def modified_z_score(intensity):
    '''Modified z score using median insted of mean'''
    median_int = np.median(intensity)
    mad_int = np.median([np.abs(intensity - median_int)])
    modified_z_scores = 0.6745 * (intensity - median_int) / mad_int
    return modified_z_scores

def fixer(y,m):
    '''Removing spikes'''
    threshold = 10 # binarization threshold. 
    spikes = abs(np.array(modified_z_score(np.diff(y)))) > threshold
    y_out = y.copy() # So we don't overwrite y
    for i in np.arange(len(spikes)):
        if spikes[i] != 0: # If we have an spike in position i
            w = np.arange(i-m,i+1+m) # we select 2 m + 1 points around our spike
            w2 = w[spikes[w] == 0] # From such interval, we choose the ones which are not spikes
            y_out[i] = np.mean(y[w2]) # and we average the value
 
    return y_out

def remap(num, oldmin, oldmax, newmin, newmax):
    """Interpolating widths information to concide with the wavelenght
    
    Parameters
    ----------
    num: str
    information for mapping
    oldmin: str
    oldmax: str
    newmin: str
    newmax:str

    
    
    """
    oldrange = oldmax - oldmin
    newrange = newmax - newmin
    return (num - oldmin)*newrange/oldrange + newmin

def dataset_creation(path_file):

    file = path_file.rsplit(sep= '/', maxsplit= 1)
    file_name = file[1].replace('.txt', '',)
    file_n = open_preprocess_file(path_file)
    file_new = saving_preprocess_file(path_file, file_n)

    headers = get_headers(file_new)
    raw_data = pd.read_csv(file_new, sep="\t",names=headers)
    df_data = pd.DataFrame(raw_data)
    df_data = df_data.drop(0)
    df_data = df_data.astype(float)

    return file_name, df_data

def file_analysis():

    path_raman = path_raman_entry.get()

    save_path = path_raman + '/Histograms/'

    contents = os.listdir(path_raman)
    files = []

    for item in contents:
        if re.search(r"\w.txt",item):
            files.append(item)
    
    
    path_file = path_raman + '/' + files[0]

    file_name, df_data = dataset_creation(path_file)

    file_spectra_window = Toplevel(root)
    root.lower()

    file_spectra_window.title("File Analysis: Raman Spectre") 
    canvas_file_spectre = Canvas(file_spectra_window, width = 1250,  height = 600) 
    canvas_file_spectre.create_image( 0, 0, image = image_file_spectre_menu, anchor = "nw") 
    canvas_file_spectre.pack(fill = "both", expand = True) 


    return

root = Tk()
root.title("Formulation Lab: Raman Spectra")

path = '/home/jessicamaldonado/OneDrive/Documents/ws_concretene/python_scripts/RamanGUI/Documents/'

image_template = PhotoImage(file= path + 'concretene_image1.png')
image_file_spectre_menu = PhotoImage(file= path + 'concretene_image2.png')

# Create Canvas 
canvas_home = Canvas(root, width = 1200,  height = 300) 
canvas_home.pack(fill = "both", expand = True) 

# Display image 
canvas_home.create_image(0, 0, image = image_template, anchor = "nw") 
  
# Add Text 
canvas_home.create_text(600, 50, text = "Welcome"
                        ,font=("Segoe UI Variable Text Semibold", 40),fill='#fff')
canvas_home.create_text(600, 90, text = "Raman Spectra Analysis Tool"
                        ,font=("Segoe UI Variable Text Semibold", 40),fill='#fff') 


path_raman_label = Label(root, text='Path', font=("Segoe UI Variable Text Semiligh", 18), background = '#fff')
path_raman_entry = Entry(root)
canvas_home.create_window(100, 200, window=path_raman_label)
canvas_home.create_window(100 + 250, 200, width = 350, window=path_raman_entry)

# Create Buttons 
button1 = Button(root, text = "Start", font=("Segoe UI Variable Text Semiligh", 18),command = file_analysis) 

# Display Buttons 
button1_canvas = canvas_home.create_window(300, 250, anchor = "nw", window = button1) 

# Execute tkinter 
root.mainloop() 