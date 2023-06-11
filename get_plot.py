import matplotlib.pyplot as plt

# set up matplotlib plot style
plt.style.use('ggplot')
# decrease font size
plt.rcParams.update({'font.size': 7})

vanilla = [17.5967,
53.7366,
51.1595,
55.0001,
54.1534,
52.1684,
57.2255,
54.0067,
58.4196,
54.4811,
56.4682,
55.8747,
54.1779,
53.2649,
52.8565,
55.0286,
52.0682,
53.8368,
53.3479,
52.9864,
56.0524,
54.5246,
54.9186,
55.3477,
56.0842,
55.0182,
]

skiplookup = [16.3981,
52.5351,
52.1985,
54.9383,
51.811,
50.3123,
53.9874,
53.531,
53.5545,
50.9516,
55.3949,
54.2889,
53.6612,
52.9043,
51.5983,
50.3625,
50.424,
54.7583,
48.2386,
51.7404,
56.356,
52.3364,
53.5734,
53.7937,
54.7532,
53.2813,
]

vocab_gumbel = [27.3376,
34.8215,
37.673,
30.3174,
34.8948,
33.0873,
36.7162,
34.4061,
39.3859,
36.9784,
33.509,
33.7688,
27.9969,
35.6353,
38.3488,
37.2604,
34.1665,
36.0762,
28.4481,
28.5977,
39.6989,
38.4121,
35.5306,
39.5799,
31.3746,
38.8703,
]

vocab_softmax = [29.0373,
36.1231,
30.5822,
43.9765,
41.9127,
38.4912,
46.3447,
44.5014,
44.2985,
36.5913,
35.8083,
42.3114,
42.5383,
36.3701,
31.2442,
38.6972,
40.4692,
45.6989,
41.051,
43.5506,
38.9295,
40.7971,
40.2469,
46.1467,
45.9028,
41.192,
]

x_axis_val = list(range(0, 251, 10))

#plot lines 
plt.plot(x_axis_val, vanilla, label='vanilla')
plt.plot(x_axis_val, skiplookup, label='skiplookup')
plt.plot(x_axis_val, vocab_softmax, label='vocab_softmax')
plt.plot(x_axis_val, vocab_gumbel, label='vocab_gumbel')
plt.legend()
# x axis label
plt.xlabel('Epochs')
# y axis label
plt.ylabel('F1 (macro))')
# plot title
plt.title('F1 vs epochs for all approaches with 10 distilled images per class')
# save plot
plt.savefig('plot.png', bbox_inches='tight', dpi=300)