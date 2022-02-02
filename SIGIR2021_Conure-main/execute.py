import test_tp_t1
import test_tp_t2
import test_tp_t3
import test_tp_t4
import test_ret_t1
import test_ret_t2
import test_ret_t3
import test_ret_t4
import os
import shutil

def main():

    case = [0]
    extention= [1]
    prune_percentage= [0.1]
    # case = [1, 2, 3]
    # extention = [1, 2, 3]
    # prune_percentage = [0.1, 0.15, 0.25, 0.3, 0.5, 0.6, 0.75]

    # dirpath = os.path.join('Results')
    # if os.path.exists(dirpath) and os.path.isdir(dirpath):
    #     shutil.rmtree(dirpath)

    for c in case:
        model = 1
        for e in extention:
            for pp in prune_percentage:
                
                # Create path
                path = 'Results' + '/Case' + str(c) + '/Model'+ str(c) + str(model)
                if not os.path.exists(path):
                    os.makedirs(path)

                # Open Case_Information file
                f = open(os.path.join(path, 'Model' + str(c) + str(model) + '_Information.txt'), "w")
                f.write('Model ' + str(c) + str(model) + '\n')
                f.write('------------------------------\n')
                f.write('Case = ' + str(c) + '\n')
                f.write('Extention = ' + str(e) + '\n')
                f.write('Prune Percentage = ' + str(pp) + '\n')
                f.close()

                # Run tests
                for test in range (1,6):
                    test_path = os.path.join(path, 'test' + str(test))
                    if not os.path.exists(test_path):
                        os.makedirs(test_path)
                    test_tp_t1.main(test_path, c, e, pp)
                    test_ret_t1.main(test_path)
                    test_tp_t2.main(test_path, c, e, pp)
                    test_ret_t2.main(test_path)
                    test_tp_t3.main(test_path, c, e, pp)
                    test_ret_t3.main(test_path)
                    test_tp_t4.main(test_path, c, e, pp)
                    test_ret_t4.main(test_path)
                    
                model += 1

if __name__ == '__main__':
    main()

