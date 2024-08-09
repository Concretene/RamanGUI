from tkinter import *
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter
from scipy.signal import find_peaks, peak_widths, peak_prominences
import seaborn as sns
import matplotlib.transforms as tr
import re
from string import ascii_letters
import sys
from pathlib import Path

user_path = os.path.expanduser('~')

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
    #rsplit_path = path.rsplit(sep='/',maxsplit=1)
    rsplit_path = path.rsplit(sep='\\',maxsplit=1)
    print(rsplit_path)
    #relative_path = rsplit_path[0] + '/process_file/'
    try:
        relative_path = rsplit_path[0] + '\\process_file\\'
        os.mkdir(relative_path)
    except:
        #print(path_folder)
        print('Already Created')
    
    file = rsplit_path[1].split(sep='.')
    file_name = file[0] + "_process." + file[1]
    #file_process_path = relative_path + "/" + file_name
    file_process_path = relative_path + "\\" + file_name
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
        replace_file_head = strip_file_head.replace("#","")
        replace_file_head = replace_file_head.replace("\t",",")
        replace_file_head = replace_file_head.replace(",,",",")
        split_file_head = replace_file_head.split(sep=",")
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

    #file = path_file.rsplit(sep= '/', maxsplit= 1)
    file = path_file.rsplit(sep= '\\', maxsplit= 1)
    file_name = file[1].replace('.txt', '',)
    file_n = open_preprocess_file(path_file)
    file_new = saving_preprocess_file(path_file, file_n)

    headers = get_headers(file_new)
    raw_data = pd.read_csv(file_new, sep="\t",names=headers)
    df_data = pd.DataFrame(raw_data)
    df_data = df_data.drop(0)
    df_data = df_data.astype(float)

    return file_name, df_data

def raman_spectre_analysis(file_name, df_data):
    j = 0
    count = 0
    ratio_full = []
    d_band = []
    g_band = []
    wave_data = []
    spectra_data = []
    data_DB = []
    data_GB = []
    data_2DB = []
    data_NB = []


    # Parameters for this case:
    l = 1000000 # smoothness
    p = 0.05 # asymmetry

    # Parameters:
    w = 5 # window (number of points)
    order = 1 # polynomial order

    ## Selection of width height
    if re.search(r"\w\dMA",file_name):
        width_height = 0.85
    else:
        width_height = 0.5

    for i, value in enumerate(df_data['Wave']):
        x_DB = []
        y_DB = []
        x_GB = []
        y_GB = []
        x_2DB = []
        y_2DB = []
        x_NB = []
        y_NB = []

        if value < (df_data['Wave'].min() + 1):
            df_signal = df_data.iloc[j:i]
            df_signal = df_signal.sort_values(by=['Wave'])

            wavelength = np.array(df_signal['Wave'])
            intensity = np.array(df_signal['Intensity'])
            despiked_spectrum = fixer(intensity, 10)
            estimated_baselined = baseline_als(intensity, l, p)
            baselined_spectrum = despiked_spectrum - np.absolute(estimated_baselined)
            count = count + 1
            #smoothed_spectrum = savgol_filter(baselined_spectrum, w, polyorder = order, deriv=0)
            smoothed_spectrum = despiked_spectrum
            for k, element in enumerate(wavelength):
                if element >= 1250 and element < 1450:
                    x_DB.append(element)
                    y_DB.append(smoothed_spectrum[k]) 
                if element >= 1480 and element < 1680:
                    x_GB.append(element)
                    y_GB.append(smoothed_spectrum[k])
                if element >= 2650 and element < 2800:
                    x_2DB.append(element)
                    y_2DB.append(smoothed_spectrum[k])
                if element >= 2800 and element < 3000:
                    x_NB.append(element)
                    y_NB.append(smoothed_spectrum[k])

            I_D = y_DB[np.argmax(y_DB)]
            I_G = y_GB[np.argmax(y_GB)]
            I_2DB = y_2DB[np.argmax(y_2DB)]
            I_N = y_NB[np.argmax(y_NB)]

            peak_DB, _ = find_peaks(y_DB, prominence= 5)
            peak_GB, _ = find_peaks(y_GB,prominence= 5)
            peak_NB, _ = find_peaks(y_NB,prominence= 5)

            prominences_DB = peak_prominences(y_DB, peak_DB)
            max_prominence_DB = peak_DB[np.argmax(prominences_DB[0])]
            prominences_GB = peak_prominences(y_GB, peak_GB)
            max_prominence_GB = peak_GB[np.argmax(prominences_GB[0])]

            peak_width_DB = peak_widths(y_DB, peak_DB, rel_height= 0.5, prominence_data= prominences_DB)
            peak_width_GB = peak_widths(y_GB, peak_GB, rel_height= width_height, prominence_data= prominences_GB)


            y_height = peak_width_DB[1][np.argmax(prominences_DB[0])]
            x_width_min = remap(peak_width_DB[2][np.argmax(prominences_DB[0])], 0, len(x_DB), x_DB[0], x_DB[-1])
            x_width_max = remap(peak_width_DB[3][np.argmax(prominences_DB[0])], 0, len(x_DB), x_DB[0], x_DB[-1])


            y_height_GB = peak_width_GB[1][np.argmax(prominences_GB[0])]
            x_width_min_GB = remap(peak_width_GB[2][np.argmax(prominences_GB[0])], 0, len(x_GB), x_GB[0], x_GB[-1])
            x_width_max_GB = remap(peak_width_GB[3][np.argmax(prominences_GB[0])], 0, len(x_GB), x_GB[0], x_GB[-1])


            if np.round(y_DB[max_prominence_DB],3) != np.round(y_DB[np.argmax(y_DB)], 3):
                I_D = y_DB[max_prominence_DB]


            if np.round(y_GB[max_prominence_GB],3) != np.round(y_GB[np.argmax(y_GB)], 3):
                I_G = y_GB[max_prominence_GB]


            ratio = I_D / I_G
            d_width = np.round(x_width_max - x_width_min, 3)
            g_width = np.round(x_width_max_GB - x_width_min_GB, 3)
            ratio_full.append(ratio)
            d_band.append(d_width)
            g_band.append(g_width)

            wave_data.append(wavelength)
            spectra_data.append(smoothed_spectrum)
            data_DB.append([x_DB, y_DB, x_DB[max_prominence_DB], y_DB[max_prominence_DB], y_height, x_width_min, x_width_max])
            data_GB.append([x_GB, y_GB, x_GB[max_prominence_GB], y_GB[max_prominence_GB], y_height_GB, x_width_min_GB, x_width_max_GB])
            data_2DB.append([x_2DB, y_2DB, x_2DB[np.argmax(y_2DB)], y_2DB[np.argmax(y_2DB)]])
            data_NB.append([x_NB, y_NB, x_NB[np.argmax(y_NB)], y_NB[np.argmax(y_NB)]])

            j = i + 1
    
    return ratio_full, g_band, d_band, wave_data, spectra_data, data_DB, data_GB, data_2DB, data_NB

def create_plot(file_name, type = 'Ratio'):
    
    sns.set_theme(style="white")



    av_ratio = np.round(np.average(ratio_full), 3)
    std_ratio = np.round(np.std(np.round(ratio_full, 4)),3)
    av_width = np.round(np.average(g_band), 3)
    std_width = np.round(np.std(g_band), 3)

    f, ax = plt.subplots(figsize=(8, 50))

    if type == 'Ratio':

        x_lim = [0, 1.5]
        y_lim = [0 , 10]
        sns.histplot(ratio_full, kde=True, alpha=0.25, kde_kws=dict(cut=10), binwidth=0.004, bins=20)
        ax.axvline(av_ratio, linestyle = 'dashed', color = 'red')
        ax.text(x_lim[1] - 0.1, y_lim[1] + 0.05,
                f'Ratio \nAverage: {av_ratio} \nSTD: {std_ratio}',
                fontsize = 12)
    
    if type == 'G band':
        x_lim = [0, 120]
        y_lim = [0 , 10]
        sns.histplot(g_band, kde=True, alpha=0.25, kde_kws=dict(cut=10), binwidth=1, bins=20)
        ax.axvline(np.average(g_band), linestyle = 'dashed', color = 'red')
        ax.text(x_lim[1] - 20, y_lim[1] + 0.1,
        f'G Band Width \n Average: {av_width}\nSTD width : {std_width}',
        fontsize = 12)
    
    
    ax.grid(True)
    ax.set_xlim(x_lim)
    ax.set_title(f'{file_name}')
    ax.set_ylim(y_lim)
    
    return f

def create_individual_plot(i, ratio_full, wave_data, spectra_data, data_DB, data_GB, data_2DB, data_NB):
    f, ax = plt.subplots(figsize=(8, 50))

    print_data = [f'Ratio: {np.round(ratio_full[i],3)}',
                  f'D band peak ->Intensity: {np.round(data_DB[i][3], 3)} Wave: {np.round(data_DB[i][2],3)}',
                  f'G band peak ->Intensity: {np.round(data_GB[i][3], 3)} Wave: {np.round(data_GB[i][2],3)}',
                  f'2DB band peak ->Intensity: {np.round(data_2DB[i][3], 3)} Wave: {np.round(data_2DB[i][2],3)}',
                  f'Width band DB: {np.round(data_DB[i][6] - data_DB[i][5], 3)}',
                  f'Width band GB: {np.round(data_GB[i][6] - data_GB[i][5], 3)}']
    
    ax.plot(wave_data[i], spectra_data[i], 'k', label='Signal')
    ax.plot(data_DB[i][0], data_DB[i][1])
    ax.plot(data_DB[i][2], data_DB[i][3], "x",color='b',label='Peaks')
    ax.hlines(data_DB[i][4], data_DB[i][5], data_DB[i][6], color = 'C3')  
    ax.plot(data_GB[i][0], data_GB[i][1])
    ax.plot(data_GB[i][2], data_GB[i][3], "x",color='b',label='Peaks')
    ax.hlines(data_GB[i][4], data_GB[i][5], data_GB[i][6], color = 'C2')
    ax.plot(data_2DB[i][0], data_2DB[i][1])
    ax.plot(data_2DB[i][2], data_2DB[i][3], "x",color='b',label='Peaks')
    ax.plot(data_NB[i][0], data_NB[i][1])
    ax.plot(data_NB[i][2], data_NB[i][3], "x",color='b',label='Peaks')
    ax.set_ylabel('Intensity')
    ax.set_xlabel('Wavelenght')
    ax.grid(True)

    return f, print_data


def individuals():
    global counter_ind

    def plot_graph(i):

        fig_ind, print_data = create_individual_plot(i, ratio_full, wave_data, spectra_data, data_DB, data_GB, data_2DB, data_NB)
        T = Text(file_individual_window, width= 55, height= 6, font=("Segoe UI Variable Text Semiligh", 12))
        canvas_file_individual.create_window(720, 670, window= T)

        for item in print_data:
            T.insert(END, item + '\n')
        
        canvas_image_ind = Canvas(canvas_file_individual, width=100, height= 100)
        canvas_file_individual.create_window(300, 100, width= 750, height=500, anchor='nw', window= canvas_image_ind)
        canvas_ind = FigureCanvasTkAgg(fig_ind, master = canvas_image_ind)   
        canvas_ind.draw() 
        toolbar_ind = NavigationToolbar2Tk(canvas_ind, canvas_image_ind)
        toolbar_ind.update()
        canvas_ind.get_tk_widget().pack()

    def individual_down():
    
        counter_ind.set(counter_ind.get() - 1)
        plot_graph(counter_ind.get())


    def individual_up():
    
        counter_ind.set(counter_ind.get() + 1)
        plot_graph(counter_ind.get())

    counter_ind = IntVar()
    counter_ind.set(0)

    file_individual_window = Toplevel(root)
    root.lower
    file_individual_window.title("Individual Raman Spectra")
    canvas_file_individual = Canvas(file_individual_window, width = 1250,  height = 740) 
    canvas_file_individual.create_image( 0, 0, image = image_file_spectre_menu, anchor = "nw")

    canvas_file_individual.create_text(100, 50, text = f"Sample Number"
                        , font=("Segoe UI Variable Text Semibold", 20), fill='black'
                        , width= 500)

    label_count = Label(file_individual_window, width=7, textvariable= counter_ind, font= ("Segoe UI Variable Text Semibold", 20))
    canvas_file_individual.create_window(150, 100, window= label_count)

    plot_graph(counter_ind.get())

    
    button_previous = Button(canvas_file_individual, text = "Previous", font=("Segoe UI Variable Text Semiligh", 18), command= individual_down)
    button_previous_canvas = canvas_file_individual.create_window(250, 650, anchor = "nw", window = button_previous)

    button_next = Button(canvas_file_individual, text = "Next", font=("Segoe UI Variable Text Semiligh", 18), command= individual_up)
    button_next_canvas = canvas_file_individual.create_window(1050, 650, anchor = "nw", window = button_next)

    canvas_file_individual.pack(fill = "both", expand = True) 


def next_figure():
    
    root.counter_file +=1 
    file_analysis()

def file_analysis():

    global path_file, files, path_raman
    global ratio_full, g_band, d_band, wave_data, spectra_data, data_DB, data_GB, data_2DB, data_NB

    def histogram_g_band():
        fig = create_plot(file_name, 'G band')
        canvas_image = Canvas(canvas_file_spectre, width=100, height= 100)
        canvas_file_spectre.create_window(50, 200, width= 750, height=450, anchor='nw', window= canvas_image)
        canvas = FigureCanvasTkAgg(fig, master = canvas_image)   
        canvas.draw() 
        toolbar = NavigationToolbar2Tk(canvas, canvas_image)
        toolbar.update()
        canvas.get_tk_widget().pack()

    def histogram_ratio():
        fig = create_plot(file_name, 'Ratio')
        canvas_image = Canvas(canvas_file_spectre, width=100, height= 100)
        canvas_file_spectre.create_window(50, 200, width= 750, height=450, anchor='nw', window= canvas_image)
        canvas = FigureCanvasTkAgg(fig, master = canvas_image)   
        canvas.draw() 
        toolbar = NavigationToolbar2Tk(canvas, canvas_image)
        toolbar.update()
        canvas.get_tk_widget().pack()

    def average_figures():

        ratio_complete = []
        g_band_complete = []

        for file in files:
            path_file = path_raman + '\\' + file
            file_name, df_data = dataset_creation(path_file)
            ratio_full, g_band, d_band, wave_data, spectra_data, data_DB, data_GB, data_2DB, data_NB = raman_spectre_analysis(file_name, df_data)
            ratio_complete.append(ratio_full)
            g_band_complete.append(g_band)

        avg_ratio_file = np.round(np.average(ratio_complete),4)
        std_ratio_file = np.round(np.std(ratio_complete),4)
        avg_g_file = np.round(np.average(g_band_complete),4)
        std_g_file = np.round(np.std(g_band_complete),4)
        canvas_file_spectre.create_text(1020, 600 
                        , text = f"Average Ratio: {avg_ratio_file}\nStd Ratio: {std_ratio_file}\nAverage G width: {avg_g_file}\nSTD G width: {std_g_file}"
                        , font=("Segoe UI Variable Text Semiligh", 18), fill='black'
                        , width= 500)


    file_spectra_window = Toplevel(root)
    root.lower()

    file_spectra_window.title("File Analysis: Raman Spectra") 
    canvas_file_spectre = Canvas(file_spectra_window, width = 1250,  height = 725) 
    canvas_file_spectre.create_image( 0, 0, image = image_file_spectre_menu, anchor = "nw") 

    path_raman = str(path_raman_entry.get())
    path_raman = path_raman.replace('"', '')
    path_raman = path_raman.replace("'",'')
    print(path_raman, type(path_raman))
    

    contents = os.listdir(path_raman)
    files = []

    for item in contents:
        if re.search(r"\w.txt",item):
            files.append(item)
    
    
    
    path_file = path_raman + '\\' + files[root.counter_file]

    file_name, df_data = dataset_creation(path_file)

    ratio_full, g_band, d_band, wave_data, spectra_data, data_DB, data_GB, data_2DB, data_NB = raman_spectre_analysis(file_name, df_data)


    
    avg_gband = np.average(g_band)
    g_band_out = [g_width for g_width in g_band if (g_width <= avg_gband * 0.4) or (g_width >= avg_gband * 1.4)]
    g_band_out.sort(reverse= True)

    fig = create_plot(file_name, 'Ratio')

    canvas_image = Canvas(canvas_file_spectre, width=100, height= 100)
    canvas_file_spectre.create_window(50, 200, width= 750, height=450, anchor='nw', window= canvas_image)
    canvas = FigureCanvasTkAgg(fig, master = canvas_image)   
    canvas.draw() 
    toolbar = NavigationToolbar2Tk(canvas, canvas_image)
    toolbar.update()

    canvas_file_spectre.create_text(220, 50, text = f"Histogram file {file_name}"
                        , font=("Segoe UI Variable Text Semibold", 20), fill='black'
                        , width= 500)
    
    canvas_file_spectre.create_text(600, 50, text = f"{root.counter_file + 1}/ {len(files)}"
                        , font=("Segoe UI Variable Text Semibold", 20), fill='black'
                        , width= 500)


    button_individual = Button(canvas_file_spectre, text = "Individual", font=("Segoe UI Variable Text Semiligh", 18), command= individuals)
    button_individual_canvas = canvas_file_spectre.create_window(950, 250, anchor = "nw", window = button_individual) 

    button_next_file = Button(canvas_file_spectre, text = "Next File", font=("Segoe UI Variable Text Semiligh", 18), command= next_figure)
    button_next_file_canvas = canvas_file_spectre.create_window(950, 350, anchor = "nw", window = button_next_file) 

    button_average = Button(canvas_file_spectre, text = "Average", font=("Segoe UI Variable Text Semiligh", 18), command= average_figures)
    button_average_canvas = canvas_file_spectre.create_window(950, 450, anchor = "nw", window = button_average)  

    button_g_band = Button(canvas_file_spectre, text = "G Band", font=("Segoe UI Variable Text Semiligh", 18), command= histogram_g_band)
    button_g_band_canvas = canvas_file_spectre.create_window(80, 130, anchor = "nw", window = button_g_band) 

    button_ratio = Button(canvas_file_spectre, text = "Ratio", font=("Segoe UI Variable Text Semiligh", 18), command= histogram_ratio)
    button_ratio_canvas = canvas_file_spectre.create_window(700, 130, anchor = "nw", window = button_ratio) 

    print(root.counter_file)
    # placing the canvas on the Tkinter window 
    canvas.get_tk_widget().pack()
    canvas_file_spectre.pack(fill = "both", expand = True) 

def exit():
    root.destroy()
    sys.exit()

root = Tk()
root.title("Formulation Lab: Raman Spectra")

root.counter_file = 0
root.protocol("WM_DELETE_WINDOW", lambda:exit())


#try:
    #concretene_path = r"\Concretene\Concretene Site - Formulation lab\Software\Raman_analysis\Documents\\"
#concretene_path = str(Path().absolute()) + '/Documents/'
concretene_path = str(Path().absolute()) + '\\Documents\\'
path_file = concretene_path
image_template = PhotoImage(file= path_file + 'concretene_image1.png')
image_file_spectre_menu = PhotoImage(file= path_file + 'concretene_image2.png')
#except:
#    concretene_path = '/OneDrive/Documents/ws_concretene/python_scripts/RamanGUI/Documents/'
#    #concretene_path = r"\OneDrive - Concretene\Formulation lab\Software\Raman_analysis\Documents\\"
#    #concretene_path = r"\OneDrive - Concretene\Formulation - backup\Documents\\"
#    #concretene_path = '/OneDrive/Formulation-backup/Documents/'
#    path_file = user_path + concretene_path
#    #image_template = PhotoImage(file=path_file + "imag\\" + "Concretene_template.png")
#    image_template = PhotoImage(file= path_file + 'concretene_image1.png')
#    image_file_spectre_menu = PhotoImage(file= path_file + 'concretene_image2.png')

#path = '/home/jessicamaldonado/OneDrive/Documents/ws_concretene/python_scripts/RamanGUI/Documents/'
#path = r"C:\Users\JessicaMaldo_p3rvdgi\OneDrive - Concretene\Documents\ws_concretene\python_scripts\RamanGUI\Documents\\"
#print(path_file)

#image_file_spectre_menu = PhotoImage(file= path + 'concretene_image2.png')

# Create Canvas 
canvas_home = Canvas(root, width = 1200,  height = 300) 
canvas_home.pack(fill = "both", expand = True) 

# Display image 
canvas_home.create_image(0, 0, image = image_template, anchor = "nw") 
  
# Add Text 
canvas_home.create_text(600, 50, text = "Welcome"
                        , font=("Segoe UI Variable Text Semibold", 40), fill='#fff')
canvas_home.create_text(600, 90, text = "Raman Spectra Analysis Tool"
                        , font=("Segoe UI Variable Text Semibold", 40), fill='#fff') 

canvas_home.create_text(120, 200, text = "Path"
                        , font=("Segoe UI Variable Text Semiligh", 18), fill='#fff')

path_raman_entry = Entry(root)
canvas_home.create_window(100 + 250, 200, width = 350, window=path_raman_entry)

# Create Buttons 
button1 = Button(root, text = "Start", font=("Segoe UI Variable Text Semiligh", 18),command = file_analysis) 
#button2 = Button(root, text = "Close", font=("Segoe UI Variable Text Semiligh", 18),command = exit) 

# Display Buttons 
button1_canvas = canvas_home.create_window(300, 220, anchor = "nw", window = button1)
#button2_canvas = canvas_home.create_window(600, 220, anchor = "nw", window = button2) 

# Execute tkinter 
root.mainloop() 