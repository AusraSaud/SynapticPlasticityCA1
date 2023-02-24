import numpy as np
from neuron_model import CA1_NeuronModel
from synapse_nmda import Synapse_NMDA
from synapse_ampa import Synapse_AMPA
from multiprocessing import Pool

class PylediPlasticity:
    
    def __init__(self, nmdabar=2e-3, ampabar=1, nmdabar_mult=1, ampabar_mult=1, nr2bbar_mult=1, w_mult=1, nmdabar_nr2cd=1e-5, save_last_record_only=False):
        self.save_last_record_only = save_last_record_only
        self.set_records_list()
        self.init_parameters()
        self.init_records()
        self.nmdabar = nmdabar * nmdabar_mult * 0.5
        self.ampabar = ampabar * ampabar_mult * 0.5
        self.stdp_range = list(range(-100, 110, 10))
        self.w_mult = w_mult

        self.neuron = CA1_NeuronModel(self.dt, save_last_record_only=save_last_record_only)
        self.synapse_nmda = Synapse_NMDA(self.dt, nmdabar=self.nmdabar, nr2bbar_mult=nr2bbar_mult, nmdabar_nr2cd=nmdabar_nr2cd)
        self.synapse_ampa = Synapse_AMPA(self.dt, ampabar=self.ampabar)
        
    def init_parameters(self):
        self.t = 0.0
        self.dt = 0.1 # ms

        self.Ip0_ampl = 20
        self.Ip0_ampl_rest = -0.5

        #Membrane parameters
        self.Vrest = -70    #mV
        self.R = 100e+6     #Ohm
        self.C = 50e-12     #F
        self.Vth = -54      #mV
        self.v = self.Vrest
        self.E_AMPA = 0
        self.E_NMDA = 0

        # AMPA
        self.gconstAMPA = 5.5e-1    # mS
        self.tauAMPA = 5            # ms

        self.tau_A = 0.5
        self.tau_B = 3.0

        self.gAMPAbar = 1.0
        self.g_ampa = 0.0
        self.synInputs = 0.0

        self.I_NMDA = 0.0
        self.I_AMPA = 0.0

        # NMDA
        self.gconstNMDA = 4e-0

        self.k = 0.33
        self.gama = 0.06 / 1e-3

        self.tau_nmda1 = 40
        self.tau_nmda2 = 0.3
        self.g_nmda_voltage = 0.0
        self.g_nmda = 0.0

        self.g_nmda_LTP = 0 # mainly on NR2B
        self.g_nmda_LTD = 0 # mainly on NR2A 


        self.gNMDAbar = 2.0
        self.gNR2Abar = 1.0
        self.gNR2Bbar = 1.0

        self.g_nr2a = 0.0
        self.g_nr2b = 0.0

        self.Alpha_nr2a	= 0.5
        self.Alpha_nr2b	= 0.1

        self.Beta_nr2a = 0.024
        self.Beta_nr2b = 0.0075

        #-----------------------------------
        #STDP
        #-----------------------------------
        
        
        self.v_trace1 = 0
        self.v_trace2 = 0
        self.v_trace3 = 0

        # Plasticity
        self.moving_thresh = 0.0
        self.moving_thresh_tau = 1.0
        self.moving_thresh_mult = 1.0

        self.moving_thresh_nmda = 0.0
        self.moving_thresh_nmda_tau = 10.0
        self.moving_thresh_nmda_mult = 1e+5

        

        self.v_trace_thresh11 = 0.2
        self.v_trace1_threshed1 = 0.0
        self.v_trace2_threshed2 = 0.0
        self.v_trace3_threshed3 = 0.0

        #Dirac 
        self.pre_dirac = 0.0
        self.dirac_trace = 0.0
        

        self.A_dirac = 1

        self.g_nmda_trace1 = 0.0
        self.g_nmda_trace2 = 0.0
        self.g_nmda_trace3 = 0.0
        
        self.k_gnmda_scaling=1e0

        

        self.nmda_thresh = 0.02
        self.nmda_threshed = 0.0

        #---------------------------------------
        #PARAMETERS
        #---------------------------------------
        
        #NMDA traces 
        self.g_nmda_tau1 = 20
        self.g_nmda_tau2 = 1000
        
        #Hill midpoint
        self.hill_ltp_mid = 11e-5
        self.hill_ltd_mid = 9e-5
       
        #Hill power
        self.hill_ltp_coef = 4
        self.hill_ltd_coef = 2
     
        #Hill moving thresholds
        #LTP threshold, inhibits LTP 
        self.moving_threshold_hill_ltp = 0.0
        self.moving_threshold_hill_ltp_multiplier_from_cd = 0   # LTP inhibition by phi_LTD
        self.moving_threshold_hill_ltp_multiplier_from_ltd = 10
        self.moving_threshold_hill_ltp_tau = 100

        #LTD AB threshold, inhibits LTD 
        self.moving_threshold_hill_ltd = 0.0
        self.moving_threshold_hill_ltd_multiplier_from_ltp = 1000   # LTD inhibition by phi_LTP
        self.moving_threshold_hill_ltd_tau = 100

        #Amplitudes
        self.A_ltp = 100e-3
        self.A_ltd = 10000e-3


        #LTD CD threshold, inhibits LTD CD from LTP 
        self.moving_threshold_hill_ltd_cd = 0.0
        self.moving_threshold_hill_ltd_multiplier_cd_from_ltp = 1000    # LTD inhibition by phi_LTP
        self.moving_threshold_hill_ltd_cd_tau = 100.0


        # Vmem Traces
        self.v_trace1_tau = 10
        self.v_trace2_tau = 10
        self.v_trace3_tau = 10

                

        #Vmem Thresholds
        self.v_trace_thresh1 = -65.0    # AB LTP
        self.v_trace_thresh2 = -67.0    # AB LTD
        self.v_trace_thresh3 = -20.0    # CD LTD

        #Dirac
        self.dirac_trace_tau = 15

        #-------------------------------
        #Not used
        #------------------------------- 
        
        #CD
        self.A_ltd_cd = 1000e-3#2000e-3 # 20e-3 #200e-3 
        self.g_nmda_tau3 = 1000                # 5 # 20   20 # 2 5000 # 20  2.0 #CD
        self.hill_ltd_mid_cd = 4e-6          # 4e-6       #1.e-6 #LTD CD 
        self.hill_ltd_cd_coef = 2

        
        #---------------------------------------------------

        self.wmin = 0.4
        self.wmax = 2.0

        self.weight = 1.0
        self.customweight = 1.0

        self.w_ltd = 0.0
        self.w_ltp = 0.0

        self.w_change = 0.0
        self.ltp_part = 0.0
        self.ltd_part = 0.0
        self.ltd_part_ab = 0.0
        self.ltd_part_cd = 0.0
        self.w_mult = 1.0
        

    #-------------------------------------------------------------------------
    # Rrecord
    #-------------------------------------------------------------------------    
    def set_records_list(self):
        self.records_list = [
            "g_ampa",
            "g_nmda",
            "g_nmda_trace1",
            "g_nmda_trace2",
            "g_nmda_trace3",
            "t",
            "v",
            "v_soma",
            "v_trace1",
            "v_trace2",
            "v_trace3",
            "v_trace1_threshed1",
            "v_trace2_threshed2",
            "v_trace3_threshed3",
            "w_ltd",
            "w_ltp",
            "ltd_part",
            "ltp_part",
            "moving_thresh",
            "moving_thresh_nmda",
            "pre_dirac",
            "dirac_trace",
            "weight",
            "hilleq_ltp",
            "hilleq_ltd",
            "hilleq_ltd_cd",
            "g_nr2a",
            "g_nr2b",
            "nmda_threshed",
            "nmda_thresh",
            "g_nmda_base",
            "moving_threshold_hill_ltd",
            "moving_threshold_hill_ltp",
            "moving_threshold_hill_ltd_cd",
            "g_nr2cd", 
        ]

    def init_records(self):
        self.records = {}
        if self.save_last_record_only:
            for record_name in self.records_list:
                self.records[record_name] = []
                self.records[record_name].append(0)
        else:
            for record_name in self.records_list:
                self.records[record_name] = []
        
    def update_records(self):
        if self.save_last_record_only:
            for record_name in self.records_list:
                self.records[record_name][-1] = getattr(self, record_name)
        else:
            for record_name in self.records_list:
                self.records[record_name].append(getattr(self, record_name))
        
    def step(self, i, syn_input=False, soma_input=False):
        Ip0_ampl = self.Ip0_ampl_rest
        if soma_input:
            Ip0_ampl = self.Ip0_ampl
        self.neuron.step(i, self.synInputs, self.I_NMDA, Ip0_ampl)
        self.synapse_ampa.step(i, self.v, syn_input)
        self.synapse_nmda.step(i, self.v, syn_input)
        self.g_ampa = self.synapse_ampa.g_ampa * self.weight * self.customweight
        self.g_nmda = self.synapse_nmda.g_nmda
        self.g_nmda_base = self.synapse_nmda.g_nmda_base
        self.g_nr2a = self.synapse_nmda.g_nr2a
        self.g_nr2b = self.synapse_nmda.g_nr2b
        self.v = self.neuron.Vdend
        self.v_soma = self.neuron.Vsoma
        
        #NMDA for LTP and LTD
        self.g_nmda_LTP = self.synapse_nmda.g_nmda_LTP
        self.g_nmda_LTD = self.synapse_nmda.g_nmda_LTD

        #AS
        self.g_nr2cd = self.synapse_nmda.g_nr2cd

        self.pre_dirac = 0.0
        if syn_input:
            self.net_receive()

        self.dynamics()
        self.update()
        self.t += self.dt
        self.update_records()


    #--------------------------------------------------------------------------------
    #PLASTICITY MODEL
    #--------------------------------------------------------------------------------

    def dynamics(self):
        self.v_trace1_threshed1 += ((max(0, (self.v - self.v_trace_thresh1)) - self.v_trace1_threshed1) / self.v_trace1_tau) * self.dt
        self.v_trace2_threshed2 += ((max(0, (self.v - self.v_trace_thresh2)) - self.v_trace2_threshed2) / self.v_trace2_tau) * self.dt
        
        self.dirac_trace += ((self.pre_dirac - self.dirac_trace) / self.dirac_trace_tau) * self.dt

        #--------------------
        #NMDA traces

        #LTP mainly NR2B
        self.g_nmda_trace1 += (((self.g_nmda_LTP) - self.g_nmda_trace1) / self.g_nmda_tau1) * self.dt
        #LTD mainly NR2A
        self.g_nmda_trace2 += (((self.g_nmda_LTD) - self.g_nmda_trace2) / self.g_nmda_tau2) * self.dt


    def net_receive(self):
        self.pre_dirac = self.A_dirac

    def update(self):
        self.I_AMPA = self.g_ampa * (self.v - self.E_AMPA)
        self.I_NMDA = self.g_nmda * (self.v - self.E_NMDA) * self.nmdabar
        self.synInputs = self.I_AMPA + self.I_NMDA

        #--------------------
        #Hill eq LTP
        #--------------------
        #LTP AB
        hill_ltp = (self.k_gnmda_scaling * self.g_nmda_trace1) ** self.hill_ltp_coef
        self.hilleq_ltp = hill_ltp / ((self.hill_ltp_mid) ** self.hill_ltp_coef + hill_ltp) - self.moving_threshold_hill_ltp
        if self.hilleq_ltp < 0:
            self.hilleq_ltp = 0
       
        
        #---------------------------------------------
        #LTP part
        #---------------------------------------------
        self.ltp_part = self.A_ltp * self.hilleq_ltp * self.v_trace1_threshed1

        #--------------------
        #Hill eq LTD AB
        #--------------------
        #LTD AB
        hill_ltd = (self.k_gnmda_scaling * self.g_nmda_trace2) ** self.hill_ltd_coef
        self.hilleq_ltd = hill_ltd / ((self.hill_ltd_mid) ** self.hill_ltd_coef + hill_ltd) - self.moving_threshold_hill_ltd
        if self.hilleq_ltd < 0:
            self.hilleq_ltd = 0

        #---------------------------------------------
        #LTD part
        #---------------------------------------------

        self.ltd_part_ab = self.A_ltd * self.hilleq_ltd  * self.v_trace2_threshed2 * self.dirac_trace

        #--------------------
        #Hill eq LTD CD
        #--------------------
        #LTD CD
        hill_ltd_cd = (self.k_gnmda_scaling * self.g_nmda_trace3) ** self.hill_ltd_cd_coef
        self.hilleq_ltd_cd = hill_ltd_cd / ((self.hill_ltd_mid_cd) ** self.hill_ltd_cd_coef + hill_ltd_cd) - self.moving_threshold_hill_ltd_cd
        if self.hilleq_ltd_cd < 0:
            self.hilleq_ltd_cd = 0

        self.ltd_part_cd = 0
        self.ltd_part = self.ltd_part_ab 
        

        #------------------------------------------------------------------------------------------
        #Hill thresholds
        #------------------------------------------------------------------------------------------
        #LTP AB threshold 
        self.moving_threshold_hill_ltp += ((-self.moving_threshold_hill_ltp + self.hilleq_ltd * self.moving_threshold_hill_ltp_multiplier_from_ltd) / self.moving_threshold_hill_ltp_tau) * self.dt
        
        #LDT AB threshold
        self.moving_threshold_hill_ltd += ((-self.moving_threshold_hill_ltd + self.hilleq_ltp * self.moving_threshold_hill_ltd_multiplier_from_ltp) / self.moving_threshold_hill_ltd_tau) * self.dt
       
        #LTD CD threshold, affected by Hill LTP
        self.moving_threshold_hill_ltd_cd =0
        
        #------------------------------------------------------------------------------------------

        ltpmult = (self.wmax - self.weight)
        ltdmult = (self.weight - self.wmin)

        self.w_change = (self.ltp_part * ltpmult - self.ltd_part * ltdmult) * self.w_mult

        self.w_ltp += self.ltp_part * ltpmult * self.w_mult * self.dt
        self.w_ltd += self.ltd_part * ltdmult * self.w_mult * self.dt
        
        self.weight += self.w_change * self.dt
        
        if self.weight < 0:
            self.weight = 0
    
    def get_record_data(self):
        return self.records

    def run_stdp_tests_static(pairings = 2, frequency = 0.5, nmdabarmult = 1, w_mult = 1, nr2bbar_mult = 1, pre_spikes = 1, post_spikes = 1, nmdabar_nr2cd = 1e-5, synparams = {}, stdp_range=range(-100, 110, 10), first_post=False, save_last_record_only=False):
        taskrange = [(pairings, frequency, stdp_dt, nmdabarmult, w_mult, nr2bbar_mult, pre_spikes, post_spikes, nmdabar_nr2cd, synparams, first_post, save_last_record_only) for stdp_dt in stdp_range]
        with Pool(len(taskrange)) as p:
            experiments = p.map(PylediPlasticity.run_single_stdp_static, taskrange)
        return experiments
        
    def run_single_stdp_static(input_tuple):
        (pairings, frequency, stdp_dt, nmdabarmult, w_mult, nr2bbar_mult, pre_spikes, post_spikes, nmdabar_nr2cd, synparams, first_post, save_last_record_only) = input_tuple
        model = PylediPlasticity(nmdabar_mult=nmdabarmult, w_mult=w_mult, nr2bbar_mult=nr2bbar_mult, nmdabar_nr2cd=nmdabar_nr2cd, save_last_record_only=save_last_record_only)

        for synparam in synparams:
            setattr(model, synparam, synparams[synparam])
            
        dt = model.dt
        pre_pair_interval = 10
        pre_pair_interval_dt = pre_pair_interval / dt
        post_pair_interval = 10
        post_pair_interval_dt = post_pair_interval / dt


        interval = int(1000/frequency)

        start = 100
        stop = interval * (pairings-1) + start + 300

        dt_int_pair = int((interval * pairings)/dt)
        dt_start_dend = int((start + 2)/dt)
        dt_start_soma = int(start/dt)
        dt_stop = int(stop/dt)    

        dt_interval = int(interval/dt)
        dt_stdp_dt = int(stdp_dt/model.dt)

        syn_inputs = []
        if first_post:
            pre_delay = 0
        else:
            pre_delay = post_pair_interval_dt * (post_spikes-1)

        for i in range(pre_spikes):
            syn_inputs += list(np.arange(dt_start_dend + pre_pair_interval_dt * i - dt_stdp_dt + pre_delay, dt_start_dend + dt_int_pair - dt_stdp_dt, dt_interval))


        soma_delta = -1.0 / dt
        soma_inputs = []
        for i in range(post_spikes):
            soma_inputs += list(np.arange(dt_start_soma + soma_delta + post_pair_interval_dt * i, dt_start_soma + soma_delta + dt_int_pair, dt_interval))

        soma_input_period_ms = 5

        soma_input_period = 0
        for i in range(dt_stop):
            syn_input_on = False
            if  soma_input_period > 0:
                soma_input_period -= 1
            if (i in syn_inputs):
                syn_input_on = True
            if (i in soma_inputs):
                soma_input_period = soma_input_period_ms / model.dt
            model.step(i, syn_input_on, soma_input_period > 0)
        
        return model.get_record_data()

    def run_freq_tests(self, frequency=5.0, pairings=2, soma_on=False, stdp_dt=10, start=300, afterstop=100, old_i = 0):
        interval = int(1000/frequency)
        start = 300
        afterstop = 100
        stop = interval * (pairings-1) + start + afterstop

        
        dt_interval = int(interval/self.dt)
        dt_start = int(start/self.dt)
        dt_stop = int(stop/self.dt)

        syn_inputs = np.arange(dt_start, dt_start + (dt_interval * pairings), dt_interval)
        soma_inputs = []
        if soma_on:
            soma_inputs = np.arange(dt_start - int(5 / self.dt) + int(stdp_dt/self.dt), dt_start - int(5 / self.dt) + int(stdp_dt/self.dt) + (dt_interval * pairings), dt_interval)

        soma_input_period_ms = 5

        soma_input_period = 0
        for i in range(dt_stop):
            syn_input_on = False
            if  soma_input_period > 0:
                soma_input_period -= 1
            if (i in syn_inputs):
                syn_input_on = True
            if (i in soma_inputs):
                soma_input_period = soma_input_period_ms / self.dt
            self.step(i+old_i, syn_input_on, soma_input_period > 0)
        self.run_data = self.get_record_data()
        return dt_stop
