import math 
import random
import time
input_logs = "input2.txt" 
cities = []
with open(input_logs,'r') as i:
    lines = i.readlines()
total = int(lines[0].replace("\n",""))
counter = 1
avg_height = 0
avg_width = 0
tot = 0
avg_length = 0
start_time = time.time()
for x in range(1, len(lines)):
    city_arr = lines[x].replace("\n","").split()
    c_height = int(city_arr[0])
    c_width = int(city_arr[1])
    c_length = int(city_arr[2])
    avg_height = avg_height + c_height
    avg_width = avg_width + c_width
    avg_length = avg_length + c_length
    cities.append([x,c_height,c_width,c_length])
    counter = counter + 1
avg_height = avg_height / total
avg_width = avg_width / total
avg_length = avg_length / total
distances = {}


quad1 = []
quad2 = []
quad3 = []
quad4 = []
quad5 = []
quad6 = []
quad7 = []
# quad1 + quad8 + quad7 + quad2 + quad3 + quad6 + quad5 + quad4
quad8 = []
for x in range(len(cities)):
    c_x = cities[x]
    if c_x[1] <= avg_height and c_x[2] <= avg_width and c_x[3] <= avg_length:
        quad1.append(c_x)
    elif c_x[1] <= avg_height and c_x[2] > avg_width and c_x[3] <= avg_length:
        quad2.append(c_x)
    elif c_x[1] <= avg_height and c_x[2] <= avg_width and c_x[3] > avg_length:
        quad4.append(c_x)
    elif c_x[1] <= avg_height and c_x[2] > avg_width and c_x[3] > avg_length:
        quad3.append(c_x)
    elif c_x[1] > avg_height and c_x[2] <= avg_width and c_x[3] <= avg_length:
        quad8.append(c_x)
    elif c_x[1] > avg_height and c_x[2] > avg_width and c_x[3] <= avg_length:
        quad7.append(c_x)
    elif c_x[1] > avg_height and c_x[2] <= avg_width and c_x[3] > avg_length:
        quad5.append(c_x)
    elif c_x[1] > avg_height and c_x[2] > avg_width and c_x[3] > avg_length:
        quad6.append(c_x)
    distances[c_x[0]]={}
    
    
    for y in range(len(cities)):
        
        if x != y:
            c_y = cities[y]
            if c_y[0] not in distances:
                distances[c_y[0]]={}
            sum = math.sqrt((c_x[1] - c_y[1])**2 + (c_x[2] - c_y[2])**2 + (c_x[3] - c_y[3])**2)
            distances[c_x[0]][c_y[0]] = sum
            distances[c_y[0]][c_x[0]] = sum

#print("total",total)
#print(cities)
#print(distances)

def generate_population(amount,cities):
    paths  = []
    for it in range (amount):

        path = list(cities)
        random.shuffle(path)
        #paths.append(path)
        path1 = list(quad1)
        random.shuffle(path1)
        path2 = list(quad2)
        random.shuffle(path2)
        path3 = list(quad3)
        random.shuffle(path3)
        path4 = list(quad4)
        random.shuffle(path4)
        path5 = list(quad5)
        random.shuffle(path5)
        path6 = list(quad6)
        random.shuffle(path6)
        path7 = list(quad7)
        random.shuffle(path7)
        path8 = list(quad8)
        random.shuffle(path8)
        # quad1 + quad8 + quad7 + quad2 + quad3 + quad6 + quad5 + quad4
        paths.append(path1+path2+path3+path4+path5+path6+path7+path8)
        paths.append(path1+path8+path7+path2+path3+path6+path5+path4)
        paths.append(path1+path8+path5+path4+path3+path6+path7+path2)
        paths.append(path2+path1+path4+path3+path6+path5+path8+path7)
        paths.append(path3+path6+path5+path4+path1+path2+path7+path8)
    return paths

def evaluate_path(path):
    total_distance = 0
    for it in range(len(path)-1):
        c_x = path[it]
        c_y = path[it+1]
        try:
            total_distance = total_distance + distances[c_x[0]][c_y[0]]
        except:
            try:
                total_distance = total_distance + distances[c_y[0]][c_x[0]]
            except:
                st3=""
                print(c_y[0])
                print(c_x[0])
                for it2 in path:
                    st3 = st3+":"+str(it2[0])
                print(st3)
                print("FAAAAAAIIIL")
                print(len(path))
                exit()
            
        
    c_x = path[len(path)-1]
    c_y = path[0]
    try:
        total_distance = total_distance + distances[c_x[0]][c_y[0]]
    except:
        total_distance = total_distance + distances[c_y[0]][c_x[0]]
    return total_distance**2

candidates = generate_population(700,cities)
#print("candidates")
#print(candidates)
results_f = {}

for it in range(len(candidates)):
    path_res = evaluate_path(candidates[it])
    #print(path_res)
    results_f[it] = 1/path_res
    tot = tot + results_f[it]
results = sorted(results_f.items(), key=lambda item: item[1], reverse=True)
the_best = results[0]
#print("Results")
#print(results)
print("tot")
def get_winner(contestants):
    winner = tot*random.random()
    acum = 0
    for it in results:
        acum = acum + it[1]
        if acum >= winner:
            return it

def mutate_array(path):
    #return path
    p_len = len(path)-1
    pos_i = random.random()*p_len
    pos_j = random.random()*p_len
    cont = path[pos_i]
    path[pos_i] = path[pos_j]
    path[pos_j] = cont
    return path


def merge_pair(pair):
    #print("Merge Pair")
    candidate1 = pair[0]
    candidate2 = pair[1]
    start_r = random.uniform(0.1,0.4)
    end_r = random.uniform(0.5,0.9)
    start_cut = int(math.floor(start_r*total))
    end_cut = int(math.floor(end_r*total))
    #print("Cuts",start_cut,end_cut)
    slice1 = candidate1[start_cut:end_cut]
    slice2 = candidate2[:start_cut]
    slice3 = candidate2[end_cut:]        
    rm_slice2 = []
    rm_slice3 = []
    for it in slice1:
        if it in slice2:
            rm_slice2.append(slice2.index(it))
            #print("slice2",it[0])
            #print("slice2",slice2[slice2.index(it)])
        if it in slice3:
            #print("slice3",it[0])
            #print("slice3",slice3[slice3.index(it)])
            rm_slice3.append(slice3.index(it))
    if len(rm_slice2)>0 or len(rm_slice3)>0:
        for it in candidate2:
            if it in slice1:
                continue
            if it in slice2:
                continue
            if it in slice3:
                continue
            if len(rm_slice2)>0:
                rm_c = rm_slice2[0]
                rm_slice2.pop(0)
                slice2[rm_c]=it
            else:
                if len(rm_slice3)>0:
                    rm_c = rm_slice3[0]
                    rm_slice3.pop(0)
                    slice3[rm_c]=it
            if len(rm_slice2)==0 and len(rm_slice3)==0:
                break
    return slice2+slice1 + slice3

def replace_last(new_guy,score):
    global results
    global tot
    replace_pos = results[len(candidates)-1][0]
    
    tot = tot - results_f[replace_pos]+score
    candidates[replace_pos] = new_guy
    results_f[replace_pos] = score
    
    results = sorted(results_f.items(), key=lambda item: item[1], reverse=True)

def replace_last_arr(guys,scores):
    global results
    global tot
    counter = 1
    for it in range(len(guys)):
        replace_pos = results[len(candidates)-(it+1)][0]
        tot = tot - results_f[replace_pos]+scores[it]
        candidates[replace_pos] = guys[it]
        results_f[replace_pos] = scores[it]
    results = sorted(results_f.items(), key=lambda item: item[1], reverse=True)
    

print(tot*random.random())
print(1/the_best[1])
score_r = 0
num_candidates = 100000
pairs = []
while score_r < 0.98:
    pn1 = get_winner(results)
    p1 = candidates[pn1[0]]
    pn2 = get_winner(results)
    p2 = candidates[pn2[0]]
    best = math.sqrt(1/results[0][1])
    worst = math.sqrt(1/results[len(candidates)-1][1])
    pair = [p1,p2]
    offspring = merge_pair(pair)
    score = 1 / (evaluate_path(offspring))
    marker = int(math.floor(len(candidates)/2))-1
    if score > results[0][1]:
        print("YAAAAAAAAAAAYYYAYAYAY")
    if score >= results[marker][1]:
        #print("Replacing")
        guys = [offspring]
        scores = [score]
        #replace_last(offspring,score)
        for x in range(10):
            marker = int(math.floor(len(candidates)/3))-1
            offspring = merge_pair(pair)
            score = 1 / (evaluate_path(offspring))
            if score >= results[marker][1]:
                guys.append(offspring)
                scores.append(score)
                #replace_last(offspring,score)
        replace_last_arr(guys,scores)
        best = math.sqrt(1/results[0][1])
        worst = math.sqrt(1/results[len(candidates)-1][1])
        #print(score)
        #print(math.sqrt(1/results[0][1]),math.sqrt(1/results[len(candidates)-1][1]),time.time()-start_time,best/worst)
        #print(results[marker][1])
        #print(results[len(candidates)-1][1])
    num_candidates = num_candidates -2
    score_r = best/worst

t_res = (1 / results[0][1])
print(math.sqrt(t_res))
print(math.sqrt(1/results[0][1]))
print(math.sqrt(1/results[len(candidates)-1][1]))
print("original",math.sqrt(1/the_best[1]))
print("time",time.time()-start_time)
winner_pos = results[0][0]
winner_path = candidates[winner_pos]
text_file = open("output.txt", "w")
for it in winner_path:
    val = str(it[1])+" "+str(it[2])+" "+str(it[3])+"\n"
    n = text_file.write(val)
it = winner_path[0]
val = str(it[1])+" "+str(it[2])+" "+str(it[3])+"\n"
n = text_file.write(val)
text_file.close()
exit()    
cand_paid = [candidates[0],candidates[1]]
res_strain = merge_pair(cand_paid)
st1 = ""
st2 = ""
st3 = ""
for it in cand_paid[0]:
    st1 = st1+":"+str(it[0])
for it in cand_paid[1]:
    st2 = st2+":"+str(it[0])
for it in res_strain:
    st3 = st3+":"+str(it[0])

print(st1)
print(st2)
print(st3)
evaluate_path(res_strain)

