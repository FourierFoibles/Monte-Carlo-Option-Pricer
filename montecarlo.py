import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

st.title('Monte Carlo Option Pricing Simulation and Black Scholes')
st.subheader('By Christopher Mattocks')

st.header('Call Option',divider='blue')
st.markdown(r'''We start with one of the simplest financial objects: the call option. Assuming the option is European,
             we have an expiry date T and a strike price K. The holder of a European call option can choose whether or not to exercise
            the option at time T to gain a payoff of Sₜ - K ( at t=T). Clearly an option holder only exercises if Sₜ > K (t=T), so the 
         payoff at T is max (Sₜ - K,0). We assume that the stock price follows a Geometric Brownian Motion, i.e. the stock price 
         Sₜ is a stochastic process satisfying the SDE:''')
st.latex(r'dS_t=σS_t dW_t +μS_t dt')
st.markdown(r' where W is a standard Brownian Motion. Given this, it is possible to obtain the Black-Scholes pricing formula for a call option: ')
st.latex( r'C = S_0 \, N(d_+) - K e^{-rT} \, N(d_-)')
st.latex(r' d_+ = \frac{\ln\left(\frac{S_0}{K}\right) + \left(r + \frac{\sigma^2}{2} \right) T}{\sigma \sqrt{T}}')
st.latex(r'd_- = d_+ - \sigma \sqrt{T}')
st.markdown('''In the following, we simulate multiple paths of the stock price (as a GBM) , calculate their payoffs, discount them backwards in time and average to obtain
         the current day price of a call option using a Monte Carlo method. We compare the Monte-Carlo Price and the BS price to assess viability of Monte-Carlo methods.
          Below are sliders for the various parameters. Note that \'steps\' refers to the increments used to split the entire interval in order to model GBM discretely -- increasing steps brings the paths closer to a real GBM. ''')
st.markdown(''' (Note here, and in most following cases, that we can calculate the value of a corresponding put option using put-call parity)''')
# parameters
N = st.slider('Number of paths', min_value=1, max_value=1000, value=100)
T = st.slider('Total time (T)', min_value=1.0, max_value=10.0, value=5.0)
vol=st.slider('Volatility(σ)', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
drift=st.slider('Drift(μ)', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
s_0=st.slider('Initial price S_0', min_value=20, max_value=20000, value=50)
steps = st.slider('Number of steps', min_value=100, max_value=1000, value=1000)
strike=st.slider('Strike price (K)', min_value=0, max_value=10000, value=1000)


dt = T / steps #small time increment

# simulate GBM paths
t = np.linspace(0, T, steps+1) # equally spaced points in time to calculate data for
dt = T / steps
Z = np.random.randn(N, steps) #creates an array of dimesnion 'N' by 'steps' of samples from a random normal
increments = (drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * Z 
log_S = np.cumsum(increments, axis=1) # sum the GBM fluctuations for each paths so we get an array of the values of each path at each timestep
log_S = np.hstack((np.zeros((N,1)), log_S))  # add initial 0 for time 0
S = s_0 * np.exp(log_S)  # exponentiate to get actual paths



total_call=0

for path in S:
    total_call=total_call+max(path[-1]-strike,0) #adding together all of the payoffs

discounted_monte_expectation=np.exp(-drift*T)*(total_call/N) #this is the Monte Carlo value for the call (taking average and discounting)
 

def bs_call(initial,strk,drf,vl,tim):
#calculate BS call value
    dee_plus=(np.log(initial/strk)+(drf+(vl**2)/2)*tim)/(vl*(tim**(1/2)))
    dee_minus=(np.log(initial/strk)+(drf-(vl**2)/2)*tim)/(vl*(tim**(1/2)))

    bs_value_fn=initial*norm.cdf(dee_plus)-strk*np.exp(-drf*tim)*norm.cdf(dee_minus)
    return bs_value_fn


# Plotting
fig, ax = plt.subplots()
for i in range(N):
    ax.plot(t, S[i],alpha=0.7)
ax.set_xlabel("Time")
ax.set_ylabel("S(t)")
ax.set_title(f"{N} Paths of GBM")
ax.axhline(y=strike, color='r', linestyle='--', label='Strike')
ax.legend()
st.pyplot(fig)


st.subheader(f'Black-Scholes value for a Call :{bs_call(s_0,strike,drift,vol,T)}')
st.subheader(f'Monte Carlo simulated value:{discounted_monte_expectation}')

st.header('Asian Options',divider='blue')
st.markdown('''Asian options are a type of financial instrument that depends on the average value of the underlying stock price, i.e. the "path" the stock takes up to
            expiry of the option. Due to the path-dependent nature of the option, there is no closed form pricing formula for Asian options. This is therefore one 
            major advantage that Monte-Carlo option pricing methods has over explicit Black-Scholes pricing. We use the same paths simulated above to calculate the value of an average price European Asian call option.
            This is an option with the only exercise opportunity being at expiry with payoff''')
st.latex('max(0,S_{avg}−K)')
st.markdown('''All of the parameters are the same as in the simulation for the vanilla call, including the strike price K.''')
st.markdown('''You will need to lower the strike price in comparison to the vanilla call option to get a 
            good simulation since the average price is nearly always significantly lower than the final price.''' )

total_asian_call=0

for path in S:
    total_asian_call=total_asian_call+max((sum(path)/len(path))-strike,0)

discounted_asian_expectation=np.exp(-drift*T)*(total_asian_call/N)

st.subheader(f'Monte Carlo simulated value of Asian Call:{discounted_asian_expectation}')




st.header('Option with Discrete Dividend',divider='blue')
st.markdown('''Now we consider an option with a discrete dividend payment at time Td. We model this as a proportional payment DS - where 
            0 < D < 1 and S- is the stock price just before dividend payment. By no arbitrage, we must have that S+ = (1-D)S- since Value after = Value before - Dividend.
            Again by no-arbitrage, we conclude that there must be no change in the value of the option at the time of the dividend payment (one could buy just before and sell 
            immediately after a dividend payment or vice versa to arbitrage if this were not the case). Thus we can once again price a European call option with a discrete dividend
            using Black-Scholes risk-neutral pricing:''')
st.latex('V^{Ddiv}(S,t)=V^{call}((1-D)S,t)')
st.markdown('for t < Td. Again we can simulate asset price paths and use Monte Carlo pricing methods to explore this option: ')

# parameters
N_d = st.slider('Number of paths', min_value=1, max_value=1000, value=100, key='number_d')
T_d = st.slider('Total time (T)', min_value=1.0, max_value=10.0, value=5.0, key='time_d')
vol_d=st.slider('Volatility(σ)', min_value=0.0, max_value=1.0, value=0.5, step=0.01, key='vol_d')
drift_d=st.slider('Drift(μ)', min_value=0.0, max_value=1.0, value=0.5, step=0.01, key='drift_d')
s_0_d=st.slider('Initial price S_0', min_value=0, max_value=100, value=50, key='s_0_d')
steps_d = st.slider('Number of steps', min_value=100, max_value=1000, value=1000, key='steps_d')
strike_d=st.slider('Strike price (K)', min_value=20, max_value=20000, value=1000, key='strike_d')
dividend_date=st.slider('Dividend date', min_value=0.0 ,max_value=float(T_d), value=float(T_d/2), key='div_date_d')
dividend_amount=st.slider('D (dividend proportion)',min_value=0.000000000001, max_value=0.9999999,value=0.5, key='div_amt_d')

dt_d = T_d / steps_d #small time increment

before_steps = math.floor(dividend_date / dt_d)
after_steps=int(steps_d-before_steps)
t_d_before = np.linspace(0, dividend_date, before_steps + 1)
t_d_after = np.linspace(dividend_date, T_d, after_steps + 1)[1:] #equally spaced points after dividend date, removing the dividend date
Z_d = np.random.randn(N_d, before_steps)
increments_d = (drift_d - 0.5 * vol_d**2) * dt_d + vol_d * np.sqrt(dt_d) * Z_d
log_S_d = np.cumsum(increments_d, axis=1)
log_S_d = np.hstack((np.zeros((N_d, 1)), log_S_d))  
S_before = s_0_d * np.exp(log_S_d)

S_at_dividend = S_before[:, -1]  # S - (last column from each row)
S_post_dividend = (1 - dividend_amount) * S_at_dividend  # S +

#after dividend
Z_after = np.random.randn(N_d, after_steps)
increments_after = ((drift_d - 0.5 * vol_d**2) * dt_d +
                    vol_d * np.sqrt(dt_d) * Z_after)
log_S_after = np.cumsum(increments_after, axis=1)
log_S_after = np.hstack((np.zeros((N_d, 1)), log_S_after))
S_after = S_post_dividend[:, None] * np.exp(log_S_after) # turns the list of post dividend values into a column vector, then multiplies/broadcasts through the 'after' fluctuations to get the 'after' paths

# combine to make full paths
S_full = np.hstack((S_before, S_after[:, 1:]))  # skip duplicated time point
t_full = np.concatenate((t_d_before, t_d_after))

total_call_d=0

for path in S_full:
    total_call_d=total_call_d+max(path[-1]-strike_d,0) #adding together all of the payoffs

discounted_monte_expectation_d=np.exp(-drift_d*T_d)*(total_call_d/N_d) #this is the Monte Carlo value for the call (taking average and discounting)
 

fig_d, ax_d = plt.subplots()
for i in range(N_d):
    ax_d.plot(t_full, S_full[i],alpha=0.7)
ax_d.set_xlabel("Time")
ax_d.set_ylabel("S(t)")
ax_d.set_title(f"{N_d} Paths of GBM")
ax_d.axhline(y=strike_d, color='r', linestyle='--', label='Strike')
ax_d.axvline(x=dividend_date, color='b', label='Dividend Payment')
ax_d.legend()
st.pyplot(fig_d)

st.subheader(f'Black-Scholes value for a Call :{bs_call((1-dividend_amount)*s_0_d,strike_d,drift_d,vol_d,T_d)}')
st.subheader(f'Monte Carlo simulated value:{discounted_monte_expectation_d}')

# now for the compound option section 

st.header('Multi-stage Options',divider='blue')
st.markdown('''Exotic options may have many different features, which can make it hard to find an explicit mathematical formula for their Black–Scholes value.
                For example, there may be barriers. If the stock price hits a barrier, a certain behaviour of the option is triggered — for example, a knock-out option
                has its value wiped to 0 if the underlying asset price falls below the barrier.''')

st.markdown('''In this example, we simulate an option with two features: a modified call-on-call option. Before the intermediate date, there is a barrier such that if
                the stock price drops below it, the option expires worthless. At the intermediate date, the holder has the right to buy a call option (or not). There is
                no barrier after the intermediate date, and if the holder chooses to exercise, the option behaves like a vanilla call after the intermediate date.''')

st.markdown("""For a standard down-and-out call option, the payoff is first extended to be zero whenever the barrier is breached. 
    This modified option can then be priced using the Black-Scholes formula. 
    By adding the value of this modified option to that of its reflected payoff (as given by the formula below), 
    we obtain the value of the original barrier option.""")

st.latex(r"R_B(F(S)) = -\left(\frac{S}{B}\right)^{1 - \frac{2r}{\sigma}} F\left(\frac{B^2}{S}\right)")

st.markdown("""We do not implement this formula here, as our focus is on Monte Carlo methods for pricing more complex barrier features.""")

st.markdown('''Note: In the following simulation, when an option is knocked out, we display the corresponding stock price path as dropping to zero. This is purely for visualisation purposes — in reality, the underlying stock price does not go to zero.
             We use this convention to indicate that the path is “dead” and will contribute nothing to the final payoff..''') 
# parameters
N_k = st.slider('Number of paths', min_value=1, max_value=1000, value=500, key='number_k')
T_k = st.slider('Total time (T)', min_value=1.0, max_value=10.0, value=2.0, key='time_k')
vol_k=st.slider('Volatility(σ)', min_value=0.0, max_value=1.0, value=0.5, step=0.01, key='vol_k')
drift_k=st.slider('Drift(μ)', min_value=0.0, max_value=1.0, value=0.5, step=0.01, key='drift_k')
knock_barrier=st.slider('Knock-out Barrier', min_value=0, max_value=100, value=50, key='knock-out')
s_0_k=st.slider('Initial price S_0', min_value=float(knock_barrier), max_value=100.0, value=(float(knock_barrier)+100.0)/2, key='s_0_k')
steps_k = st.slider('Number of steps', min_value=100, max_value=1000, value=1000, key='steps_k')
strike_k=st.slider('Strike price (K)', min_value=20, max_value=2000, value=105, key='strike_k')
intermediate_date=st.slider('Intermediate date', min_value=0.0 ,max_value=float(T_k), value=float(T_k/2), key='intermediate_date')
under_strike_k=st.slider('Underlying strike price', min_value=20, max_value=500, value=150, key='under_strike_k')


dt_k= T_k/steps_k

# simulate GBM paths as before
t_k = np.linspace(0, T_k, steps_k+1) 
Z_k = np.random.randn(N_k, steps_k) 
increments_k = (drift_k - 0.5 * vol_k**2) * dt_k + vol_k * np.sqrt(dt_k) * Z_k 
log_S_k = np.cumsum(increments_k, axis=1) 
log_S_k = np.hstack((np.zeros((N_k,1)), log_S_k)) 
S_k = s_0_k * np.exp(log_S_k)  

#knockout options which wouldn't be exercised at intermediate date to receive the call
intermediate_step_indx= math.floor(intermediate_date/dt_k)
for path in S_k:
    price_t_1=path[intermediate_step_indx]
    if bs_call(price_t_1,strike_k,drift_k,vol_k,T_k-intermediate_date)<=under_strike_k:
        path[intermediate_step_indx:] = 0

#now knockout options which hit the barrier before intermediate time:
for path in S_k:
    path_before_intmed=path[:intermediate_step_indx]
    if any(x<knock_barrier for x in path_before_intmed):
        barrier_idx = None
        for i, x in enumerate(path_before_intmed):
            if x < knock_barrier:
                barrier_idx = i
                break
        path[barrier_idx:]=0

total_call_k=0

for path in S_k:
    total_call_k=total_call_k+max(path[-1]-strike_k,0)

discounted_monte_expectation_k=np.exp(-drift_k*T_k)*(total_call_k/N_k)
 
fig_k, ax_k = plt.subplots()

for i in range(N_k):
    ax_k.plot(t_k, S_k[i], alpha=0.7)
ax_k.plot(t_k[intermediate_step_indx:], [under_strike_k]*(len(t_k)-intermediate_step_indx),
          color='red', linestyle='--', label='Underlying strike')
ax_k.plot(t_k[:intermediate_step_indx+1], [strike_k]*(intermediate_step_indx+1),
          color='purple', linestyle='--', label='Initial call strike')
ax_k.plot(t_k[:intermediate_step_indx+1], [knock_barrier]*(intermediate_step_indx+1),
          color='black', label='Knock-out barrier')
ax_k.axvline(x=intermediate_date, color='blue', linestyle='--', label='Intermediate date')
ax_k.set_xlabel("Time")
ax_k.set_ylabel("S(t)")
ax_k.set_title(f"{N_k} Paths of GBM")
ax_k.legend()
st.pyplot(fig_k)

st.subheader(f'Monte-Carlo value for this option :{discounted_monte_expectation_k}')

st.header('Conclusion', divider='blue')

st.markdown("""We have presented a brief exposition of Monte Carlo methods for pricing financial options. 
    In particular, we have shown that when the explicit Black-Scholes pricing formula is known, 
    Monte Carlo methods yield results very close to the actual option value. Moreover, for options with more complex features—such as path dependence (Asian options), multistage behaviour, or barriers— 
    Monte Carlo pricing provides a practical way to determine fair values without relying on explicit solutions of the Black-Scholes PDE.""")
