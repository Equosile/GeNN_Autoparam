####################################
#          PyGeNN Project          #
#      by Prof Thomas Nowotny      #
#####################################
#     HYPERPARAMETER AUTOMATION     #
#                FOR                #
#     AGENT CLASSIFICATION TEST     #
#####################################
#
# Institute: University of Sussex
# Supervisor:
#             Prof Christopher L. Buckley
#             Dr   Paul Kinghorn
#
# Script Writer: Equosile (jk488@sussex.ac.uk)
# https://github.com/Equosile/GeNN_Autoparam
#
#
#
#
###################################################################################
#                                                                                 # 
# The BSD Zero Clause License (0BSD)                                              #
#                                                                                 #
# Copyright (c) 2023 Equosile.                                                    #
#                                                                                 #
# Permission to use, copy, modify, and/or distribute this software for any        #
# purpose with or without fee is hereby granted.                                  #
#                                                                                 #
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH   #
# REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY     #
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,    #
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM     #
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR   #
# OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR          #
# PERFORMANCE OF THIS SOFTWARE.                                                   #
#                                                                                 #
###################################################################################
#
#
#
#

# ESSENTIAL LIBRARIES
from pygenn.genn_model import (GeNNModel, init_connectivity, create_cmlf_class,
                                create_custom_sparse_connect_init_snippet_class)
from pygenn.genn_wrapper import NO_DELAY
import pygenn.genn_model as genn_model

import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt

# FOR LOCAL-SYSTEM-USE
import os

# ETC
import copy
import math
import random

# FOR BENCHMARK AND LOGGING
import time
import datetime


b_backend_t = datetime.datetime.now()
benchmark_starting_time = time.strftime("%y-%m-%d %H:%M:%S")
print("Starting Time: \n\t\t", benchmark_starting_time)

################
#   OS MACRO   #
################
# Directory Separator '/' versus '\'
# Linux & Windows
dir_mk = '/'

# Windows
# dir_mk = "\\"

############################
#     GLOBAL LIBRARIES     #
############################

# THE NUMBER OF DETECTION NEURON (eKC, output neuron, etc.) POPULATIONS
    # p.s. SYMBOLISM of DETECTABLE OBJECTS 0 TO 3
    #       INDEX 0 == DULLNESS (Nothing Found)
    #       INDEX 1 == UNKNOWNNESS (Something Found)
    #       INDEX 2 == RUBBISH (Not an Interest)
    #       INDEX 3 == FOOD (Mission Object)
_NDN = 4

# FROM THE MODEL DEFINITIONS...
# HOW MUCH LONG THE STUDY OF SNNs
int_parsing_step = 50
float_exposure = 25.0

# KCDN OBSERVER BIN SIZE OF THE HISTOGRAM
kcdn_hist_bin = np.arange( -0.004, 0.2, 0.004 )

# MNIST DATASET SHOULD BE SITUATED IN THE SAME WORKING DIRECTORY
filepath = os.getcwd()
print("Current Working Directory: \n\t{0}".format(filepath))

# BUFFER OFFSET FOR READING DATASET
int_frame_size = 28
int_img_ignored_byte = 16
int_lb_ignored_byte = 8

str_dataset_path = filepath + dir_mk + "dataset" + dir_mk + "dataset.csv"
# DATASET EXTRACTOR ( Dataset Format: Vector ( <- PNG ) )
def dataset_loader( int_size=80 ):
    #global str_dataset_path

    max_item = int_size
    print( "Dataset Size: ", max_item )
    if max_item > 0:
        data = loadtxt( str_dataset_path, delimiter=',' )
        return data
    else:
        print( "Dataset_Load_Error: Data-size should be larger than 0." )
        return None

# RESULT IMAGES WILL BE SAVED IN THIS PATH
str_result_path = filepath + dir_mk + "report"
if not os.path.isdir(str_result_path):
    os.makedirs(str_result_path)

log_code = time.time()
int_logcode = int(log_code)
str_logcode = str(int_logcode)

log_filename = filepath + dir_mk + str_logcode + "_log.txt"
log_f = open(log_filename, 'w')   
def logwriter( str_fpt ):
    if isinstance(str_fpt, str):
        for obj_str in str_fpt:
            log_f.write(obj_str)
        log_f.write('\n')


# ADDITIONAL REGULATION TOWARD THE IMAGE INPUT, LESSENING THE INTENSITY OF EACH PIXEL
#
# INPUT  : numpy_Array, size_of_Unit
# OUTPUT : numpy_Array
def normaliser( vec_input, int_unit, bool_debug=False ):
    int_size_input = len( vec_input )
    if int_unit <= 0:
        int_unit = int_size_input
    if bool_debug:
        logwriter( "UNIT_SIZE: " + str( int_unit ) )
    int_size_iteration = int( int_size_input / int_unit )
    int_sqr_unit = int( math.sqrt( int_unit ) )
    
    int_signal = 0
    float_intensity = 0
    int_average = 0
    np_out = np.zeros( 0 )
    
    int_index = 0
    for int_iteration_j in range( 0, int_size_iteration ):
        np_temp = np.zeros(0)
        for int_iteration_i in range( 0, int_unit ):
            if vec_input[ int_index ] > 0:
                int_signal = int_signal + 1
                float_intensity = float_intensity + vec_input[ int_index ]
                
            np_temp = np.concatenate( ( np_temp, vec_input[ int_index ] ), axis=None )
            int_index = int_index + 1
            
        str_log = str( int_iteration_j ) + "=Original: "
        if bool_debug:
            logwriter( str_log )
            logwriter( str( np_temp ) )
        
        if int_signal != 0:
            int_average = int( float_intensity / int_signal )
            np_out = np.concatenate( ( np_out, np_temp / int_average ), axis=None )
        else:
            np_out = np.concatenate( ( np_out, np_temp ), axis=None )
            
        int_print_indicator = int_index - int_unit
        for index_j in range( 0, int_sqr_unit ):
            list_temp = list()
            for index_i in range( int_print_indicator, int_print_indicator + int_sqr_unit ):
                list_temp.append( np_out[ index_i ] )
                int_print_indicator = index_i + 1
            if bool_debug:
                logwriter( str( list_temp ) )
            
        int_signal = 0
        float_intensity = 0
        int_average = 0
        
    
    return np_out


###########################
#    PyGeNN PROCESSING    #
###########################

# GLOBAL POPULATION SET FOR TEACHING SIGNAL
#np_DN = np.arange(_NDN)

# Label Floating-Point To Integer Classification
def label_tag( label ):
    int_case = _NDN
    
    if -1 < label and label < 1:
        #print("Case: DULLNESS")
        int_case = 0     
    elif 0 < label and label < 2:
        #print("Case: UNKNOWNNESS")
        int_case = 1
    elif 1 < label and label < 3:
        #print("Case: RUBBISH")
        int_case = 2
    elif 2 < label and label < 4:
        #print("Case: FOOD")
        int_case = 3
    else:
        #print("Undefined Case")
        int_case = _NDN
    
    return int_case

# TEACHING SIGNAL CONTROLLER
def teacher_signal( label, intensity ):
    #global np_DN
    
    np_DN = np.arange(_NDN)
    size_out = np_DN.size
    int_tag = -1
    right_intensity = intensity
    inverted_intensity = (-1) * intensity
    
    if isinstance(label, int):
        int_tag = label
    else:
        int_tag = label_tag( label )
    
    for int_index in range(0, size_out):
        if int_index == int_tag:
            np_DN[ int_index ] = right_intensity
        else:
            np_DN[ int_index ] = inverted_intensity
    
    return np_DN

# DURING "model.step_time()"
def feedTool( neuron_population, model_timestep ):
    neuron_population.pull_current_spikes_from_device()
    
    int_length = np.copy( neuron_population.spike_count ).flatten()
    list_spikes = list()
    if int_length != 0:
        list_spikes.append( np.ones( int_length ) * model_timestep )
        list_spikes.append( np.copy( neuron_population.current_spikes ) )
    else:
        list_dummy = list()
        list_spikes.append(list_dummy)
        list_spikes.append(list_dummy)
    return list_spikes

# AFTER the iteration of "model.step_time()"
def viewTool( model, list_spikes, int_milestone, str_name="TEST" ):
    #global str_result_path
    
    list_time = list()
    list_id = list()
    str_milestone = str( int_milestone )
    float_axis_min = ( int_milestone / model.dT ) - ( int_parsing_step / model.dT )
    float_axis_max = float_axis_min + ( float_exposure / model.dT )
    list_axis_limit = [ float_axis_min, float_axis_max ]
    
    for list_i in list_spikes:
        list_time.append(list_i[0])
        list_id.append(list_i[1])
    kc_spike_t = np.hstack(list_time)
    kc_spike_id = np.hstack(list_id)
    
    size_plot = 0.1
    if len(kc_spike_id) < (_NDN + 1):
        size_plot = 3.0
    
    axA = plt.figure(str_name)
    plt.scatter(kc_spike_t, kc_spike_id, s=size_plot)
    plt.xlim( list_axis_limit )
    
    filename = str_result_path + dir_mk + str_milestone + "_" + str_name + ".png"
    plt.savefig(filename)
    plt.close("all")

# AFTER the iteration of "model.step_time()", Reporting Detection Neurons
# Additionally "FITNESS SCORE" Generator for Genetic Algorithm
def detectionTool( model, list_spikes, int_milestone, int_label, str_name="TEST" ):
    #global str_result_path
    
    ga_fitness = 0
    
    current_class = int_label
    list_time = list()
    list_id = list()
    str_milestone = str( int_milestone )
    float_axis_max = int_milestone / model.dT
    float_axis_min = float_axis_max - ( int_parsing_step / model.dT )
    list_axis_limit = [ float_axis_min, float_axis_max ]
    
    for list_i in list_spikes:
        list_time.append( list_i[ 0 ] )
        list_id.append( list_i[ 1 ] )
    dn_spike_t = np.hstack( list_time )
    dn_spike_id = np.hstack( list_id )
    
    size_plot = 0.1
    if len( dn_spike_id ) < ( _NDN + 1 ):
        size_plot = 3.0
    
    axA = plt.figure( str_name )
    plt.scatter( dn_spike_t, dn_spike_id, s=size_plot )
    
    temp_index = 0
    for temp_x, temp_y in zip( dn_spike_t, dn_spike_id ):
        temp_index = temp_index + 1
        str_index = str( temp_index )
        plt.text( temp_x, temp_y, str_index, ha="center", va="center", color='r' )
        if temp_index == 1:
            plt.text( temp_x, temp_y, str( temp_x ), ha="right", va="top", color='b' )
            if int( temp_y ) == current_class:
                ga_fitness = ga_fitness + 1
        elif temp_index == 2:
            plt.text( temp_x, temp_y, str( temp_x ), ha="right", va="top", color='g' )
            if int( temp_y ) == current_class:
                ga_fitness = ga_fitness + 1
        
        #CASTING INTEGER FROM NP.FLOAT
        int_temp_y = label_tag( temp_y )
    
    plt.xlim( list_axis_limit )
    
    filename = str_result_path + dir_mk + str_milestone + "_" + str_name + ".png"
    plt.savefig(filename)
    plt.close("all")
    
    return ga_fitness

# NO TIME TO DRAW SUCH A SLOW ILLUSTRATION PROCESS
def dT_speed( model, list_spikes, int_milestone, int_label ):

    ga_fitness = 0
    
    current_class = int_label
    list_time = list()
    list_id = list()
    
    for list_i in list_spikes:
        list_time.append( list_i[ 0 ] )
        list_id.append( list_i[ 1 ] )
    dn_spike_t = np.hstack( list_time )
    dn_spike_id = np.hstack( list_id )
    
    size_plot = 0.1
    if len( dn_spike_id ) < ( _NDN + 1 ):
        size_plot = 3.0
    
    temp_index = 0
    for temp_x, temp_y in zip( dn_spike_t, dn_spike_id ):
        temp_index = temp_index + 1
        str_index = str( temp_index )
        if temp_index == 1:
            if int( temp_y ) == current_class:
                ga_fitness = ga_fitness + 1
        elif temp_index == 2:
            if int( temp_y ) == current_class:
                ga_fitness = ga_fitness + 1
        
        #CASTING INTEGER FROM NP.FLOAT
        int_temp_y = label_tag( temp_y )
    
    return ga_fitness

# Spiking Neural Network Record Simulation
# Running the Model By Corresponding Time-Steps
# Check Up Global Variables As For the Result
#spike_t = list()
#spike_id = list()
#fT_iKC = list()
#fT_eKC = list()

def parse_spiking_neural_network( model, PNs, \
                                    np_iKCs, np_eKCs, \
                                    fT_iKC, fT_eKC, \
                                    spike_t, spike_id, milestone ):
    # Initialising Global Variables
    #global spike_t
    #global spike_id
    #global fT_iKC
    #global fT_eKC
    
    #global float_exposure
    
    while ( milestone - model.t ) > float_exposure:
        model.step_time()
        
        PNs.pull_current_spikes_from_device()
        n = len( PNs.current_spikes )
        if ( n != 0 ):
            spike_t.append( np.ones( n ) * model.timestep )
            tmp_spikes = np.copy( PNs.current_spikes )
            spike_id.append( tmp_spikes )
            
        #np_iKCs
        temp_iKC = list()
        temp_iKC = feedTool(np_iKCs, model.timestep)
        #temp_iKC[0]: Spike-Time
        #temp_iKC[1]: Spike
        fT_iKC.append(temp_iKC)
        #np_eKCs
        temp_eKC = list()
        temp_eKC = feedTool(np_eKCs, model.timestep)
        #temp_eKC[0]: Spike-Time
        #temp_eKC[1]: Spike
        fT_eKC.append(temp_eKC)
    
    # LETTING GeNN MODEL TO BE STABLE
    # From this stage, the GeNN Model is exposed by empty stimulus (constant 0s).
    PNs.extra_global_params["offset"].view[:] = 0
    
    while model.t < milestone:
        model.step_time()
        
        PNs.pull_current_spikes_from_device()
        n = len( PNs.current_spikes )
        if ( n != 0 ):
            spike_t.append( np.ones( n ) * model.timestep )
            tmp_spikes = np.copy( PNs.current_spikes )
            spike_id.append( tmp_spikes )
        
        #np_iKCs
        temp_iKC = list()
        temp_iKC = feedTool(np_iKCs, model.timestep)
        #temp_iKC[0]: Spike-Time
        #temp_iKC[1]: Spike
        fT_iKC[0] = [ *fT_iKC[0], *temp_iKC[0] ]
        fT_iKC[1] = [ *fT_iKC[1], *temp_iKC[1] ]
        #np_eKCs
        temp_eKC = list()
        temp_eKC = feedTool(np_eKCs, model.timestep)
        #temp_eKC[0]: Spike-Time
        #temp_eKC[1]: Spike
        fT_eKC[0] = [ *fT_eKC[0], *temp_eKC[0] ]
        fT_eKC[1] = [ *fT_eKC[1], *temp_eKC[1] ]

# RELEASING PREVIOUS MEMORIES TO NULL
#def empty_global_buffers():
#    #global spike_t
#    #global spike_id
#    #global fT_iKC
#    #global fT_eKC
#    
#    spike_t = list()
#    spike_id = list()
#    fT_iKC = list()
#    fT_eKC = list()



#################
#    M A I N    #
#################
#
# ARGUMENTS:
#            GENOTYPES == HYPERPARAMETERS:
#            {
#                 the size of mushroombody == _NKC
#                 learning curves == tau, alpha(aPlus & aMinus), wmax, offset
#                 teacher signal == float_intensity
#            } 
#
#
# RETURN:
#         FITNESS SCORE FOR Genetic Algorithm
#
def main( geno_a, geno_b, geno_c, geno_d, geno_e, geno_f, fastpace=False ):
    norm_a = int( geno_a )
    norm_b = round( geno_b, 4 )
    norm_c = round( geno_c, 8 )
    norm_d = round( geno_d, 4 )
    norm_e = round( geno_e, 8 )
    norm_f = round( geno_f, 2 )

    #global spike_t
    #global spike_id
    #global fT_iKC
    #global fT_eKC
    spike_t = list()
    spike_id = list()
    fT_iKC = list()
    fT_eKC = list()
    
    #global int_parsing_step
    
    
    # RETURN
    ga_total_fitness = 0
    
    ########################
    # TEST SIZE DEFINITION #
    ########################

    # THE NUMBER OF ANTENNAL LOBE NEURON POPULATIONS
    # p.s. NEURON INDEX FROM 0 TO 99
    _NAL = 100

    # THE NUMBER OF KENYON CELL(MUSHROOM BODY) NEURON POPULATIONS
    _NKC = norm_a


    #############################
    #    GeNN HyperParameter    #
    ############################# 

    s_PNKC_ini = {"g": 0.0033555} # Conductance of Scalar

    ps_PNKC_p = {"tau": 1.0, # Decay time constant [ms] 
                 "E": 0.0} # Reversal potential [mV]

    PN_para = {"tspike": 0.2, 
               "trefract": 0.3, 
               "Vspike": 80.0,
               "Vrest": -60.0}
              
    PN_ini = {"V": -60.0,
              "spikeTime": -1e5}

    p = {"gNa": 7.15,   # Na conductance in [muS]
         "ENa": 50.0,   # Na equi potential [mV]
         "gK": 1.43,    # K conductance in [muS]
         "EK": -95.0,   # K equi potential [mV] 
         "gl": 0.02672, # leak conductance [muS]
         "El": -63.563, # El: leak equi potential in mV, 
         "C": 0.143}    # membr. capacity density in nF

    ini = {"V": -60.0,      # membrane potential
           "m": 0.0529324,  # prob. for Na channel activation
           "h": 0.3176767,  # prob. for not Na channel blocking
           "n": 0.5961207}  # prob. for K channel activation

    vec_conductance = list()
    dist_random = random.Random()
    for int_i in range(_NKC * _NDN):
        #Normal Distribution; gauss( mean, std_deviation )
        vec_conductance.append(dist_random.gauss(0.00001, 0.000005))
    s_KCDN_ini = {"g": vec_conductance} # Conductance of Scalar

    ps_KCDN_p = {"tau": 10.0, # Decay time constant [ms] 
                 "E": 0.0} # Reversal potential [mV]

    learn_p = {
        "tauPlus": norm_b,
        "tauMinus": norm_b,
        "aPlus": norm_c,
        "aMinus": norm_c,
        "wMin": 0,
        "wMax": norm_d,
        "offset": norm_e}
        
    GGN_PARAMS = {
        "Vthresh": 40.0}

    if_init = {
        "V": 0.0}
    
    ############################
    #        GeNN Model        #
    ############################

    # GLOBAL PyGeNN MODEL == Mushroom-body Simulation
    model = GeNNModel("float", "MBodySim")
    model.dT = 0.1

    
    # Loading Dataset. The number of the dataset is 80.
    int_dataset_size = 80
    sample_dataset = dataset_loader(int_dataset_size)
    
    # Entire Dataset Size: 80
    int_ex_size = int_dataset_size
    
    # Class-Set: Total 4 Classes ('0' ~ '3')
    np_labels = np.zeros(int_ex_size)
    
    # Projection Neuron Population Processing
    # n_PN: PN Buffer Size = 28 * 28
    # The first image (28*28 size vector) is blank for spiking control.
    n_PN = int_frame_size * int_frame_size
    input_stimuli = np.zeros(n_PN * (int_ex_size + 1))
    
    # The first blank image for spiking control.
    vec_dataset = np.zeros(n_PN)
    vec_img = np.zeros(n_PN)
    np.copyto(vec_dataset, vec_img)
    
    # Label Rotation: dull, unknown, food, food, dull, unknown, scrap, scrap
    # ref. ./dataset/out
    label_rotation = [ 0, 1, 3, 3, 0, 1, 2, 2 ]
    for int_index in range(0, int_ex_size):
        vec_img = sample_dataset[ int_index ]
        vec_dataset = np.concatenate((vec_dataset, vec_img))
        np_labels[int_index] = label_rotation[int_index % 8]

    np.copyto(input_stimuli, vec_dataset)
    
    
    #########################
    #    PyGeNN Sequence    #
    #########################
    
    # Proprocessing as Normalisation
    input_stimuli = normaliser(input_stimuli, n_PN)
    
    # For weakening image stimuli
    image_modulus = 235.0
    
    # Initialisation of the PyGeNN Model
    PNs = model.add_neuron_population("PN", n_PN, "Poisson", PN_para, PN_ini)
    PNs.set_extra_global_param("firingProb", input_stimuli/image_modulus)
    # The Initial Offset is from 28*28 because Offset 0 starts from the blank.
    PNs.set_extra_global_param("offset", n_PN)
    
    np_iKCs = model.add_neuron_population("MB", _NKC, "TraubMiles", p, ini)
    np_eKCs = model.add_neuron_population("DN", _NDN, "TraubMiles", p, ini)
    
    # GGN
    #
    # Integrate & Fire (Simple Not Leaky)
    # GGN (Giant GABAergic Neuron) Mechanism
    # sim_code: "below add incoming synaptic current to voltage"
    # threshold_condition_code: "when threshold reached it emits a spike"
    # reset_code: "then it is set to 0"
    if_model = genn_model.create_custom_neuron_class(
        "IF",
        param_names=["Vthresh"],
        var_name_types=[("V","scalar")],
        sim_code=
        """
        $(V)+=$(Isyn);
        """,
        threshold_condition_code=
        """
        $(V)>=$(Vthresh)
        """,
        reset_code=
        """
        $(V)=0.0;
        """
    )
    #
    # One Inhibitory Neuron in the Population Connected to Kennyon Cells
    np_GGN = model.add_neuron_population("GGN", 1, if_model, GGN_PARAMS, if_init)
    #
    
    # Custom Weight Update Model
    # For the Synapse Connectivity Between MB and DN
    stdp_additive_model = genn_model.create_custom_weight_update_class(
        "stdp_additive",
        param_names=["tauPlus","tauMinus","aPlus","aMinus","wMin","wMax","offset"],
        var_name_types=[("g","scalar")],
        sim_code="""
            $(addToInSyn,$(g));
            const scalar dt=$(t)-$(sT_post);
            if(dt>0){
                const scalar timing=exp(-dt/$(tauMinus))-$(offset);
                const scalar newWeight=$(g)+($(aMinus)*timing);
                $(g)=fmax($(wMin),fmin($(wMax),newWeight));
            }
            """,
        learn_post_code="""
            const scalar dt=$(t)-$(sT_pre);
            if(dt>0){
                const scalar timing=exp(-dt/$(tauPlus))-$(offset);
                const scalar newWeight=$(g)+($(aPlus)*timing);
                $(g)=fmax($(wMin),fmin($(wMax),newWeight));
            }   
            """,
        is_pre_spike_time_required=True,
        is_post_spike_time_required=True)
    
    ######################
    # Synapse Population #
    ######################
    #Arguments:
    #   pop_name, matrix_type, delaly_steps,
    #   source, target,
    #   update_model, wu_param_space, wu_var_space, pre_var_space, post_var_space
    #   postsyn_model, ps_param_space, ps_var_space,
    #   connectivity_initialiser

    #PN(AL) -> KC(MB) (Arbitrary Connectivity)
    PNKC = model.add_synapse_population("AL2MB", "SPARSE_GLOBALG", NO_DELAY,
        PNs, np_iKCs,
        "StaticPulse", {}, s_PNKC_ini, {}, {},
        "ExpCond", ps_PNKC_p, {},
        init_connectivity("FixedProbability", {"prob": 0.2}))

    #MB -> DN (All-to-All)
    #"StaticPulse": Fixed Conductances
    #stdp_additive_model: STDP (Versatile Conductances)
    KCDN = model.add_synapse_population("MB2OUT", "DENSE_INDIVIDUALG", NO_DELAY,
        np_iKCs, np_eKCs,
        stdp_additive_model, learn_p, s_KCDN_ini, {}, {},
        "ExpCond", ps_KCDN_p, {})

    #KC (MB) -> GGN (All-to-All)
    # Recurrent Connectivity 1
    # DeltaCurr: "delta current where all current dumped into neuron in one single timestep"
    KCGGN = model.add_synapse_population("KC2GGN", "DENSE_GLOBALG", NO_DELAY,
        np_iKCs, np_GGN,
        "StaticPulse", {}, {"g": 1.0}, {}, {},
        "DeltaCurr", {}, {})

    #GGN -> KC (MB) (All-to-All)
    # Recurrent Connectivity 2
    # ExpCurr: "the exponential synapse spreads out the inhibition for a few timesteps"
    GGNKC = model.add_synapse_population("GGN2KC", "DENSE_GLOBALG", NO_DELAY,
        np_GGN, np_iKCs,
        "StaticPulse", {}, {"g": -5.0}, {}, {},
        "ExpCurr", {"tau": 20.0}, {})

    #######################################################    
    # Teacher Model (Extra Conductance to Output Neurons) #
    #######################################################
    teacher_model = genn_model.create_custom_current_source_class(
        "current_inject",
        var_name_types=[("magnitude", "scalar")],
        injection_code="$(injectCurrent,$(magnitude));"
    )

    # intensity: How strong the teaching signal is.
    float_intensity = 0.0
    teacher = model.add_current_source(
        "teacher", teacher_model, np_eKCs,
        {}, {"magnitude":float_intensity})

    # iteration: (Integer) the Number of Iterations
    # Currently, Hints for the Label Index
    int_iteration = 0
    
    # Current Working Label: sample_lb[int_iteration]
    lb_class = label_tag(np_labels[int_iteration])

    # Parsing the Model Data, Transferring them to GPU
    model.build()
    model.load()
    
    str_mb_size = "Mushroom Body Size: " + str(_NKC)
    
    if not fastpace:
        logwriter(str_mb_size)
    
    # Depending on the current "Class (=label)",
    # Teaching Signal guides the model.
    #
    # E.g.
    #teacher_signal(teacher, sample_lb[ int_iteration ])
    #teacher.vars["magnitude"].view[:] = np_DN
    #
    # However, the Manipulation of Teaching signals
    # can be feasible ONLY AFTER the model is actually built. 
    new_intensity = norm_f
    teacher.vars["magnitude"].view[:] = teacher_signal(lb_class, new_intensity)
    teacher.push_var_to_device("magnitude")
    # "Vthresh"
    # In order to set a proper threshold for GGN, the study of spikings at MB is required.
    #np_GGN.vars["Vthresh"].view[:] = ???
    
    # How much long the one phrase is set here.
    # E.g. if int_parsing_step is 50 then the image is being exposed for 50 MilliSeconds.
    #int_parsing_step = 50
    total_simulation_length = int_parsing_step * int_ex_size
    
    # Tracking Weight-values in KCDN for studying the brain plasticity
    ########################################
    # Example Debugging:                   #
    #                                      #
    #print(KCDN.vars)                      #
    #print(dir(teacher.vars["magnitude"])) #
    #print(teacher.vars["magnitude"].view) #
    #                                      #
    # Find out 'g' for the Weight-value.   #
    #time.sleep(5)                         #
    ########################################
    # X-axis: Labels (or classes, tags, etc.)
    #np_KCDN_log_x = np.zeros(int_ex_size)
    #np_KCDN_log_x[int_iteration] = lb_class
    list_KCDN_log_x_a = list()
    list_KCDN_log_x_a.append(lb_class)
    # Y-axis: Weight-values (or 'g', conductance degree, etc.)
    #np_KCDN_log_y = np.zeros(int_ex_size)
    KCDN.pull_var_from_device("g")
    node_value = KCDN.vars['g'].view.copy()
    #np_KCDN_log_y[int_iteration] = node_value.copy()
    list_KCDN_log_y_a = list()
    list_KCDN_log_y_a.append(node_value)
    str_buff = "[0]KCDN: " + str(node_value)
    
    if not fastpace:
        logwriter(str_buff)
    
    ga_interim_fit_kcdn = 0
    if len( node_value ) > 0:
        eval_kcdn = np.mean( node_value )
        if eval_kcdn != 0:
            if not math.isnan( eval_kcdn ):
                ga_interim_fit_kcdn = 1
    ga_total_fitness = ga_total_fitness + ga_interim_fit_kcdn
    
    # Tracking "V" values in MB for studying the GGN Threshold
    list_MB_log_x_a = list()
    list_MB_log_x_a.append(lb_class)
    list_MB_log_y_a = list()
    np_iKCs.pull_var_from_device("V")
    population_value = np_iKCs.vars["V"].view.copy()
    list_MB_log_y_a.append(population_value)
    str_buff = "[0]MB: " + str(population_value)
    
    if not fastpace:
        logwriter(str_buff)
    
    ga_interim_fit_mb = 0
    if len( population_value ) > 0:
        eval_mb = np.mean(population_value)
        if eval_mb != 0:
            if not math.isnan( eval_mb ):
                ga_interim_fit_mb = 0.5
    ga_total_fitness = ga_total_fitness + ga_interim_fit_mb
    
    # milestone: (Integer) Total Progress Indicator
    int_milestone = 0
    
    # offset: (Integer) Position of Input Stream
    int_offset = n_PN
    
    while int_milestone < total_simulation_length:   
        int_milestone = int_milestone + int_parsing_step
        str_milestone = str(int_milestone)
        
        # Running the Model by timing steps
        # Look up global variables above after this sequence.
        parse_spiking_neural_network(model, PNs, \
                                        np_iKCs, np_eKCs, \
                                        fT_iKC, fT_eKC, \
                                        spike_t, spike_id, int_milestone)
        
        int_stack_size = len(spike_t)
        if int_stack_size <= 0:
            print("INDEX No. {0}: no image is detected!".format(int_milestone))
            print("Error Offset: ", int_offset)
        
        spike_t = np.hstack(spike_t)
        spike_id = np.hstack(spike_id)
        
        if not fastpace:
            fx, list_plt = plt.subplots(4, 1)

            str_test_A = str_milestone + "_A_spike"
            list_plt[0].set_title(str_test_A)
            list_plt[0].scatter(spike_t, spike_id, s=0.1)

            str_test_B = str_milestone + "_B_uniq_id="
            id = np.unique(spike_id)
            str_tag = str(len(id))
            str_test_B = str_test_B + str_tag
            x = np.zeros(28*28)
            x[id] = 1
            list_plt[1].set_title(str_test_B)
            list_plt[1].imshow(np.reshape(x,(28,28)))

            str_test_C = str_milestone + "_C_hist"
            x2= np.histogram(spike_id, bins= np.array(np.arange(0,28*28+1)-0.5), \
                            range=(-0.5,255.5))
            list_plt[2].set_title(str_test_C)
            list_plt[2].imshow(np.reshape(x2[0],(28,28)))

            str_test_D = str_milestone + "_D_bar"
            list_plt[3].set_title(str_test_D)
            list_plt[3].bar(x2[1][:-1],x2[0])
            
            fx.tight_layout(pad=0.0)
            str_test_set = str_result_path + dir_mk + str_milestone + "_t_SNN.png"
            plt.savefig(str_test_set)
            plt.close("all")
        
        #MUSHROOM-BODY EXAMINATION
        str_test_vA = "v_" + str_milestone + "A_iKC"
        
        if not fastpace:
            viewTool(model, fT_iKC, int_milestone, str_test_vA)
        #OUTPUT NEURON EXAMINATION
        str_test_vB = "v_" + str_milestone + "B_eKC"
        
        ga_interim_fit_dt = 0
        if fastpace:
            ga_interim_fit_dt = dT_speed(model, fT_eKC, \
                          int_milestone, \
                          lb_class)
        else:
            ga_interim_fit_dt = detectionTool(model, fT_eKC, \
                          int_milestone, \
                          lb_class, \
                          str_name=str_test_vB)
        
        ga_total_fitness = ga_total_fitness + ga_interim_fit_dt
        
        # Clear the currently existing data
        #empty_global_buffers()
        spike_t = list()
        spike_id = list()
        fT_iKC = list()
        fT_eKC = list()
        
        int_iteration = int_iteration + 1
        if int_iteration < int_ex_size:
            # Current Working Label: sample_lb[int_iteration]
            lb_class = label_tag(np_labels[int_iteration])
        
            # Move the input offset to one tick (28*28) right side
            int_offset = int_offset + n_PN
            PNs.extra_global_params["offset"].view[:] = int_offset
            
            # Tracking the brain plasticity -> Looking up weight-values ('g') in KCDN
            # Logging it via text files too along with pyplot.
            list_KCDN_log_x_a.append(lb_class)
            KCDN.pull_var_from_device("g")
            node_value = KCDN.vars['g'].view.copy()
            list_KCDN_log_y_a.append(node_value)
            str_buff = "[" + str(int_iteration) + "]KCDN: " + str(node_value)
            
            if not fastpace:
                logwriter(str_buff)
                
            ga_interim_fit_kcdn = 0
            if len( node_value ) > 0:
                eval_kcdn = np.mean( node_value )
                if eval_kcdn != 0:
                    if not math.isnan( eval_kcdn ):
                        ga_interim_fit_kcdn = 0.5
            ga_total_fitness = ga_total_fitness + ga_interim_fit_kcdn
            
            # Debugging "Vthresh" of GGN
            list_MB_log_x_a.append(lb_class)
            np_iKCs.pull_var_from_device("V")
            population_value = np_iKCs.vars["V"].view.copy()
            list_MB_log_y_a.append(population_value)
            str_buff = "[" + str(int_iteration) + "]MB: " + str(population_value)
            
            if not fastpace:
                logwriter(str_buff)
                
            ga_interim_fit_mb = 0
            if len( population_value ) > 0:
                eval_mb = np.mean(population_value)
                if eval_mb != 0:
                    if not math.isnan( eval_mb ):
                        ga_interim_fit_mb = 0.5
            ga_total_fitness = ga_total_fitness + ga_interim_fit_mb
        
            # Updating Teaching Signals
            teacher.vars["magnitude"].view[:] = teacher_signal(lb_class, new_intensity)
            teacher.push_var_to_device("magnitude")
            
            # Feedback UI
            int_feedback_epoch = int(int_ex_size / 10)
            if int_feedback_epoch == 0:
                int_feedback_epoch = 1
            if int_iteration % int_feedback_epoch == 0:
                print("{0}: Processing...".format(int_iteration))

    
    # Finalising the Stage...
    print("Current milestone: {0} / (total) {1}".format( \
                    int_milestone, \
                    total_simulation_length))

    # End of Function
    a_backend_t = datetime.datetime.now()
    benchmark_ending_time = time.strftime("%y-%m-%d %H:%M:%S")
    print("Ending Time: \n\t\t", benchmark_ending_time)
    delta_backend_t = a_backend_t - b_backend_t
    print("Elapsed Time: ", delta_backend_t)
    
    if not fastpace:
        logwriter("Starting Time: ")
        logwriter(str(benchmark_starting_time))
        logwriter("Ending Time: ")
        logwriter(str(benchmark_ending_time))
        logwriter("Delta: ")
        logwriter(str(delta_backend_t))
    
    print("END OF Function")
    # POSITIVE POINT FOR NO ERROR
    ga_total_fitness = ga_total_fitness + 0.5
    
    return ga_total_fitness
    


##################################################################################################
##################################################################################################
#
###############################
#        GA SIMULATION        #
############################################
#   SEARCHING FOR DECENT HYPERPARAMETERS   #
#          FOR PyGeNN AGENT BRAIN          #
############################################
#
def rand_genotype_generator():
    _mb = np.random.randint( 2200 )
    
    _tau = np.random.randint( 100 ) * np.random.rand()
    
    _alpha = np.random.rand() / np.random.randint( low=1, high=1000000 )
    
    _wmax = np.random.rand()
    
    _offs = np.random.rand() / np.random.randint( low=1, high=1000000 )
    
    _teach = np.random.rand() * np.random.randint( low=1, high=3 )
    
    return [ _mb, _tau, _alpha, _wmax, _offs, _teach ]

def rand_individual_type_generator( int_type ):
    if int_type == 0:
        _mb = np.random.randint( 2200 )
        
        return _mb
        
    elif int_type == 1:
        _tau = np.random.randint( 100 ) * np.random.rand()
        
        return _tau
        
    elif int_type == 2:
        _alpha = np.random.rand() / np.random.randint( low=1, high=1000000 )
        
        return _alpha
        
    elif int_type == 3:
        _wmax = np.random.rand()
        
        return _wmax
        
    elif int_type == 4:
        _offs = np.random.rand() / np.random.randint( low=1, high=1000000 )
        
        return _offs
        
    elif int_type == 5:
        _teach = np.random.rand() * np.random.randint( low=1, high=3 )
        
        return _teach


def _header( lst_test ):
    return lst_test[0]



if __name__ == "__main__":
    #genotype_test = [ 800, 26, 0.00006, 0.4, 0.0044, 1.0 ]
    #ga_fit_guid = main( genotype_test[0], genotype_test[1], genotype_test[2], \
    #        genotype_test[3], genotype_test[4], genotype_test[5], True )

    #print( "GUIDED FITNESS SCORE:", ga_fit_guid )
    
    
    # PROBLEM SIZE == NUMBER OF ARGRUMENTS == HYPERPARAMETERS == GENOTYPES
    # No. 1 = MB SIZE
    # No. 2 = tau
    # No. 3 = alpha
    # No. 4 = wMax
    # no. 5 = offset
    # no. 6 = teacher signal
    int_problem_size = 6
    
    # TOTAL SIMULATION SIZE = batch_size * population_size
    # ONE SIMULATION DURATION = AVERAGE 2 MINUTES (DEPENDING ON CPUs)
    # ENTIRE SIMULATION DURATION (MINUTES) = 2 * batch * population
    int_batch_size = 10
    int_population_size = 10
    int_simulation_size = int_batch_size * int_population_size

    # GA PROPERTIES
    fitness_total = np.zeros( int_batch_size * int_population_size )
    
    # INITIAL POPULATION
    lst_mat_population = list( range( int_population_size ) )
    
    lst_generation_hash = list()
    fitness_generation = np.zeros( int_population_size )
    
    np_mat_population = np.zeros( ( int_population_size, int_problem_size ) )
    for int_i in range( int_population_size ):
        print( "Creating initial population... {0} / {1}".format( int_i, int_population_size ) )
        print( "Agent Definition:" )
        np_mat_population[ int_i ] = rand_genotype_generator()
        arg_a = int( np_mat_population[ int_i ][ 0 ] )
        print( "\t MUSHROOM-BODY =", arg_a )
        arg_b = round( np_mat_population[ int_i ][ 1 ], 4 )
        print( "\t tau =", arg_b )
        arg_c = round( np_mat_population[ int_i ][ 2 ], 8 )
        print( "\t alpha =", arg_c )
        arg_d = round( np_mat_population[ int_i ][ 3 ], 4 )
        print( "\t wMax =", arg_d )
        arg_e = round( np_mat_population[ int_i ][ 4 ], 8 )
        print( "\t LEARNING-CURVE-OFFSET =", arg_e )
        arg_f = round( np_mat_population[ int_i ][ 5 ], 2 )
        print( "\t TEACHER-SIGNAL-MAGNITUDE =", arg_f )
        tmp_fit = main( arg_a, arg_b, arg_c, arg_d, arg_e, arg_f, True )
        tmp_list = list()
        tmp_list.append( tmp_fit )
        lst_generation_hash.append( tmp_list )
        lst_generation_hash[ int_i ].append( np_mat_population[ int_i ] )
    
    lst_ranked_gen = sorted( lst_generation_hash, key=_header, reverse=True )
    lst_mat_population = lst_ranked_gen.copy()
    
    # BATCH-GA
    int_index_sim = 0
    for int_batch_ongoing in range( int_batch_size ):
        for int_pop_ongoing in range( int_population_size ):
            print( "Original_Pop:\n", lst_mat_population )
            comp_index_A, comp_index_B  = random.sample( range( int_population_size ), 2 )
            competitor_A = lst_mat_population[ comp_index_A ]
            competitor_B = lst_mat_population[ comp_index_B ]
            print( "competitor_A:", competitor_A )
            print( "competitor_B:", competitor_B )
            winner = -1
            loser = -2
            if competitor_A[0] > competitor_B[0]:
                winner = comp_index_A
                loser = comp_index_B
            else:
                winner = comp_index_B
                loser = comp_index_A
            np_tmp_W = lst_mat_population[ winner ][ 1 ].copy()
            np_tmp_L = lst_mat_population[ loser ][ 1 ].copy()
            
            # UNIFORM-CROSSOVER (50% FIXED-CHANCE)
            int_index_a, int_index_b, int_index_c = random.sample( range( int_problem_size ), 3 )
            np_tmp_L[ int_index_a ] = np_tmp_W[ int_index_a ]
            np_tmp_L[ int_index_b ] = np_tmp_W[ int_index_b ]
            np_tmp_L[ int_index_c ] = np_tmp_W[ int_index_c ]
            
            # MUTATION
            lst_index_m = random.sample( range( int_problem_size ), 1 )
            int_index_m = lst_index_m[ 0 ]
            print("Mutated Index     :", int_index_m)
            new_param = rand_individual_type_generator( int_index_m )
            print("Mutated Parameter :", new_param)
            np_tmp_L[ int_index_m ] = new_param
            
            # Re-ARCHIVE THE POPULATION
            arg_a = int( np_tmp_L[ 0 ] )
            print( "MUSHROOM-BODY:", arg_a )
            arg_b = round( np_tmp_L[ 1 ], 4 )
            print( "tau:", arg_b )
            arg_c = round( np_tmp_L[ 2 ], 8 )
            print( "alpha:", arg_c )
            arg_d = round( np_tmp_L[ 3 ], 4 )
            print( "wMax:", arg_d )
            arg_e = round( np_tmp_L[ 4 ], 8 )
            print( "LEARNING-CURVE-OFFSET:", arg_e )
            arg_f = round( np_tmp_L[ 5 ], 2 )
            print( "TEACHER-SIGNAL-MAGNITUDE:", arg_f )
            new_fit = main( arg_a, arg_b, arg_c, arg_d, arg_e, arg_f, True )
            lst_mat_population[ loser ][ 0 ] = new_fit
            lst_mat_population[ loser ][ 1 ] = np_tmp_L
            
            # Re-ARRANGE THE POPULATION
            lst_ranked_hash = copy.deepcopy( lst_mat_population )
            lst_ranked_hash = sorted( lst_ranked_hash, key=_header, reverse=True )
            fitness_total[ int_index_sim ] = lst_ranked_hash[0][0]
            lst_mat_population = copy.deepcopy( lst_ranked_hash )
            
            # SIMULATION-STEP-INDICATOR
            print( "Simulation Processing... Fit={0} ( {1} / {2} )".format( \
                        fitness_total[int_index_sim], int_index_sim, int_simulation_size ) )
            int_index_sim = int_index_sim + 1
            
    
    # END OF PROGRAMME
    print( "FINAL FITNESS SCORE :", fitness_total[ -1 ] )
    hyperparam = lst_mat_population[ 0 ][ 1 ]
    print( "FINAL HYPERPARAM-SET:\n" )
    arg_a = int( hyperparam[ 0 ] )
    print( "\t MUSHROOM-BODY =", arg_a )
    arg_b = round( hyperparam[ 1 ], 4 )
    print( "\t tau =", arg_b )
    arg_c = round( hyperparam[ 2 ], 8 )
    print( "\t alpha =", arg_c )
    arg_d = round( hyperparam[ 3 ], 4 )
    print( "\t wMax =", arg_d )
    arg_e = round( hyperparam[ 4 ], 8 )
    print( "\t LEARNING-CURVE-OFFSET = {0:.7f}".format( arg_e ) )
    arg_f = round( hyperparam[ 5 ], 2 )
    print( "\t TEACHER-SIGNAL-MAGNITUDE =", arg_f )
    
    str_nextline = "========================================================"
    logwriter( str_nextline )
    str_nextline = "Fintness Score : " + str( fitness_total[ -1 ] )
    logwriter( str_nextline )
    str_nextline = "HyperParameters: " + str( hyperparam )
    logwriter( str_nextline )
    str_nextline = "========================================================"
    logwriter( str_nextline )
    
    final_fit = main( arg_a, arg_b, arg_c, arg_d, arg_e, arg_f, False )
    
    log_f.close()
    
    fig = plt.figure( figsize=(19.2, 10.8), dpi=300 )
    axA = fig.add_subplot( 1, 1, 1 )
    str_title = "Fitness Tendency"
    axA.set_title( str_title )
    xs = np.arange( len( fitness_total ) )
    axA.plot( xs, fitness_total )
    fig.tight_layout( pad=5.0 )
    str_file_png = filepath + "/fitness_tendency.png"
    fig.savefig( str_file_png )

#
#
#    
# EOF
    