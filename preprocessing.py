import os
import csv
import math
import sys  # Import sys module to use sys.argv

def roundup(var):
    return float(format(var, '.6f'))

def main(dir_path, output_dir):
    files = os.listdir(dir_path)
    
    for file_name in files:
        with open(os.path.join(dir_path, file_name), 'r') as textfile:
            new_file = open(os.path.join(output_dir, file_name), 'w+')
            new_list = []

            prev_yield = 0.0
            diff_yield = 0.0
            avg_yield = 0.0
            num_moving_avg = 50
            volatile_avg_yield = 0.0
            num_volatile = 10
            curr_volatility_yield = 0.0

            for count, row in enumerate(reversed(list(csv.reader(textfile)))):
                if not count:
                    try:
                        row[8] = prev_yield
                    except Exception as e:
                        row.append(prev_yield)
                else:
                    diff_yield = roundup(float(row[7]) - prev_yield)
                    try:
                        row[8] = diff_yield
                    except Exception as e:
                        row.append(diff_yield)

                if count < num_moving_avg:
                    avg_yield = roundup((count * avg_yield + float(row[7])) / (count + 1))
                else:
                    avg_yield = roundup((num_moving_avg * avg_yield + float(row[7]) - float(new_list[count - num_moving_avg][7])) / num_moving_avg)

                prev_yield = float(row[7])

                if count < num_volatile:
                    volatile_avg_yield = roundup((count * volatile_avg_yield + float(row[7])) / (count + 1))
                else:
                    volatile_avg_yield = roundup((num_volatile * volatile_avg_yield + float(row[7]) - float(new_list[count - num_volatile][7])) / num_volatile)

                if count:
                    loop_count = min(count, num_volatile)

                    for i in range(loop_count):
                        curr_volatility_yield += math.pow((float(row[7]) - volatile_avg_yield), 2)

                    curr_volatility_yield = roundup(math.sqrt(curr_volatility_yield / loop_count))

                try:
                    row[9] = avg_yield
                    row[10] = curr_volatility_yield
                except Exception as e:
                    row.append(avg_yield)
                    row.append(curr_volatility_yield)

                new_list.append(row)
                curr_volatility_yield = 0.0

            new_list.insert(0, ['Rain Fall (mm)', 'Fertilizer(urea) (kg/acre)', 'Temperature (°C)', 'Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)', 'Yeild (Q/acre)', 'prev_day_diff', '50_day_moving_avg', '10_day_volatility'])

            writer = csv.writer(new_file)
            writer.writerows(new_list)
            new_file.close()
        textfile.close()

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
