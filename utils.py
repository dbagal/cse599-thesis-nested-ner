from torch.utils import data


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

        for i,data_row in enumerate(dataset):
            
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

        table += f'{"="*(table_width)}\n'

        return table


    @staticmethod
    def print_in_tabular_format(dataset, headers, include_serial_numbers = True,
                                col_guetter = 4, serial_num_heading = "Sr.No", 
                                table_header=None):

        print(PrettyPrint.get_tabular_formatted_string(dataset, headers, include_serial_numbers,
                                col_guetter, serial_num_heading, table_header))
