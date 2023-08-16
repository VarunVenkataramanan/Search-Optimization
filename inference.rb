require 'pycall'
PyCall.sys.path.append('/workspace')
file = PyCall.import_module("python_scripts")

pipe = file.setup()

print "Enter your query: "

query = gets.chomp

starting = Time.now

result = file.inference(pipe,query)
puts "comparison: \t#{result[0]}"
puts "price: \t\t#{result[1]}"
puts "comp: \t\t#{result[2]}"

ending = Time.now
elapsed = ending - starting
puts "time: \t\t#{elapsed}"