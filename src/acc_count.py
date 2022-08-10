
def FullTags(tags):
    state = 0
    count = 0
    for tag in tags:
        if state == 0:
            if tag[0] == "S":
                count += 1
                state = 0
            elif tag[0] == "B":
                state = 1
            else:
                state = 0
        elif state == 1:
            if tag[0] == "I":
                state = 1
            elif tag[0] == "E":
                count += 1
                state = 0
            else:
                state = 0
    return count

def acc(all_prediction, all_labels):
    assert len(all_prediction) == len(all_labels)

    intersect = list()
    for i in range(len(all_labels)):
        if all_labels[i] == all_prediction[i]: intersect.append(all_labels[i])
        else: intersect.append("O")
        
    TP = FullTags(intersect)
    total = FullTags(all_labels)
    return TP, total
    
if __name__ == "__main__":
    all_prediction = "B-Org E-Sym O O O B-Tim E-Tim O B-Sym I-Sym S-Tim B-Tim E-Tim O S-Tim O B-Sym I-Sym E-Sym O O E-Sym".split()
    all_labels = "B-Org E-Org O O O B-Tim E-Tim O B-Sym I-Sym E-Sym B-Tim E-Tim O S-Tim O B-Sym I-Sym E-Sym O B-Sym E-Sym".split()
    TP, total = acc(all_prediction, all_labels)
    print("TP: ", TP)
    print("total: ", total)