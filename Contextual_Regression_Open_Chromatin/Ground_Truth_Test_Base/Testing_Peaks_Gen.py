from Function import *

total_data = int(sys.argv[1])
noise_level = float(sys.argv[2])

each_seq_size = seq_len

generated_data, generated_weights = gen_data(each_seq_size, max_feature_signal, noise_level, total_data)

print "max value"
print np.amax(generated_data)
print "max weight"
print np.amax(generated_weights)

write_data(generated_data, "generated_data")
write_data(generated_weights, "generated_weights")

