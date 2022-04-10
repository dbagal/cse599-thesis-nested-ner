def get_csv(metrics, class_names):
    file = f"metric,{','.join(class_names)}"
    for metric, vals in metrics.items():
        file += f"\n{metric},"
        if type(vals)==list:
            file += f"{','.join([str(val) for val in vals])}"
        else:
            file += f"{vals}"
    return file
    
        
def pretty_print_results(metrics, class_names):
    nl = len(class_names)
    if metrics:
        headers = ['metric',] + class_names
        
        dataset = []
        for key, val in metrics.items():
            if type(val) == list and len(val)==nl:
                dataset += [[key]+val]
            else:
                dataset += [[key]+[val]+[0,]*(nl-1)]

        file = PrettyPrint.get_tabular_formatted_string(
                    dataset=dataset, 
                    headers=headers,
                    include_serial_numbers=False,
                    table_header="Evaluation metrics",
                    partitions=[5,9]
                )
        return file
    return None



class PrettyPrint():

    @staticmethod
    def get_tabular_formatted_string(dataset, headers, include_serial_numbers = True,
                                col_guetter = 4, serial_num_heading = "Sr.No", 
                                table_header=None, partitions=None):
        class InconsistentDataAndHeaderError(Exception):
            def __init__(self):
                msg = f"Inconsistency in number of columns and number of headers!"
                super().__init__(msg)

        class NonPrimitiveDataError(Exception):
            def __init__(self, val):
                msg = f"Dataset field value ({val}) should have primitive types (string, integer, boolean)"
                super().__init__(msg)

        column_lens = []
        num_columns = len(headers)
        num_rows = len(dataset)

        for data_row in dataset:
            if len(data_row)!=num_columns:
                raise InconsistentDataAndHeaderError()

        if include_serial_numbers:
            num_columns += 1
            headers = [serial_num_heading] + headers
            dataset = [[i+1]+row for i,row in enumerate(dataset)]

        # find maximum column length for every column 
        for col_num in range(num_columns):
            max_col_len = 0
            for row_num in range(num_rows):
                # raise error on non-primitive data field values
                if dataset[row_num][col_num]!= None and type(dataset[row_num][col_num]) not in (str, int, bool, float):
                    raise NonPrimitiveDataError(dataset[row_num][col_num])
                
                # convert all primitive values to string
                dataset[row_num][col_num] = str(dataset[row_num][col_num]) 
                max_col_len = max(max_col_len, len(dataset[row_num][col_num]), len(headers[col_num]))

            column_lens += [max_col_len]

        # print headers
        row = "|"+ " "*(col_guetter//2)
        for col_len in column_lens:
            row += "{:<"+str(col_len+col_guetter//2)+"}|" + " "*(col_guetter//2)
        
        formatted_header = row.format(*headers)

        table_width = len(formatted_header) - (col_guetter//2)

        table = f"""\n{"="*(table_width)}\n"""

        if table_header:
            table += f'{"|"+table_header.center(table_width-2)+"|"}\n{"|"+"-"*(table_width-2)+"|"}\n'
        
        table += f'{formatted_header}\n{"|"+"="*(table_width-2)+"|"}\n'

        partitions.sort()
        partitions = [row_num -i for i,row_num in zip(range(0,len(partitions)), partitions)]
        i=0
        while i< len(dataset):
            data_row = dataset[i]
            
            if i+1 not in partitions:
                row = "|"+ " "*(col_guetter//2)
                for col_len in column_lens:
                    row += "{:<"+str(col_len+col_guetter//2)+"}|" + " "*(col_guetter//2)

                table += row.format(*data_row)+"\n"
            else:
                row = "|"
                partition_row = ["-"*(col_len+2*(col_guetter//2)) for col_len in column_lens]
                for col_len in column_lens:
                    row += "{}|"
                table += row.format(*partition_row)+"\n"
                partitions.remove(i+1)
                i-=1
            i+=1

        table += f'{"="*(table_width)}\n'

        return table


    @staticmethod
    def print_in_tabular_format(dataset, headers, include_serial_numbers = True,
                                col_guetter = 4, serial_num_heading = "Sr.No", 
                                table_header=None):

        print(PrettyPrint.get_tabular_formatted_string(dataset, headers, include_serial_numbers,
                                col_guetter, serial_num_heading, table_header))
