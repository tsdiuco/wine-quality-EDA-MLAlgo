import mysklearn.myutils as myutils
"""MyPyTable class for working with csv data
"""
import copy
import csv
#from tabulate import tabulate # uncomment if you want to use the pretty_print() method
# install tabulate with: pip install tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    # def pretty_print(self):
    #     """Prints the table in a nicely formatted grid structure.
    #     """
    #     print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.column_names)

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        col_index = col_identifier
        if isinstance(col_identifier, str):
            col_index = self.column_names.index(col_identifier)
        if include_missing_values:
            return [row[col_index] for row in self.data]
        else:
            return [row[col_index] for row in self.data if row[col_index]!="NA"]

    def replace_column(self, column_name, new_column):
        col_index = self.column_names.index(column_name)
        new_data = []
        for i in range(len(self.data)):
            row = self.data[i]
            row[col_index] = new_column[i]
            new_data.append(row)
        self.data = new_data

    def get_normalized_columns(self, column_list):
        for col in column_list:
            self.replace_column(col, myutils.normalize(self.get_column(col)))
        return self.get_columns(column_list)

    def get_columns(self, column_list):
        """Extracts a list of columns from the table data as a list of lists.

        Args:
            column_list(list of str): list of column names to extract

        Returns:
            list of list of obj: 2D list of values in the columns

        Notes:
            Raise ValueError on invalid column_list
        """
        col_indexes = []
        for col in column_list:
            col_indexes.append(self.column_names.index(col))
        return [[self.data[i][j] for j in col_indexes] for i in range(len(self.data))]

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                try:
                    self.data[i][j] = float(self.data[i][j])
                except:
                    pass


    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        new_table = []
        for i in range(len(self.data)):
            if i not in row_indexes_to_drop:
                new_table.append(self.data[i])
        self.data = new_table

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        file = open(filename, 'r')
        filereader = csv.reader(file)
        self.column_names = next(filereader)
        self.data = []
        for row in filereader:
            self.data.append(row)
        self.convert_to_numeric()
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        with open(filename, 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(self.column_names)
            for row in self.data:
                writer.writerow(row)

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        new_data = []
        dupes = []
        for i in range(len(self.data)):
            row_keys = []
            for key in key_column_names:
                key_index = self.column_names.index(key)
                row_keys.append(self.data[i][key_index])
            if row_keys in new_data:
                dupes.append(i)
            else:
                new_data.append(row_keys)
        return dupes

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        new_data = []
        for row in self.data:
            if "NA" not in row:
                new_data.append(row)
        self.data = new_data

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        total = 0
        length = 0
        col_index = self.column_names.index(col_name)
        for row in self.data:
            if not isinstance(row[col_index], str):
                total += row[col_index]
                length += 1
        original_average = total / length
        for i in range(len(self.data)):
            if self.data[i][col_index] == "NA":
                self.data[i][col_index] = original_average

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.

        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]
        """
        summary_header = ["attribute", "min", "max", "mid", "avg", "median"]
        summary_data = []
        for col in col_names:
            col_data = self.get_column(col, include_missing_values=False)
            col_data = [val for val in col_data if not isinstance(val, str)]
            if len(col_data) == 0:
                continue
            summary = [col]
            summary.append(min(col_data))
            summary.append(max(col_data))
            mid = min(col_data) + ((max(col_data) - min(col_data)) / 2)
            summary.append(mid)
            summary.append(sum(col_data)/len(col_data))
            col_data.sort()
            middle = len(col_data) // 2
            median = (col_data[middle] + col_data[~middle]) / 2
            summary.append(median)
            summary_data.append(summary)
        return MyPyTable(summary_header, summary_data)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        self_key_column_indices = [self.column_names.index(key) for key in key_column_names]
        other_key_column_indices = [other_table.column_names.index(key) for key in key_column_names]
        new_header = self.column_names
        other_non_key_columns = [name for name in other_table.column_names if name not in key_column_names]
        new_header += other_non_key_columns
        new_table = []
        for i in range(len(self.data)):
            for j in range(len(other_table.data)):
                keys_match = True
                for key in range(len(self_key_column_indices)):
                    if self.data[i][self_key_column_indices[key]] != other_table.data[j][other_key_column_indices[key]]:
                        keys_match = False
                if keys_match:
                    new_row = []
                    for col in range(len(self.data[i])):
                        new_row.append(self.data[i][col])
                    for col in range(len(other_table.data[j])):
                        if col not in other_key_column_indices:
                            new_row.append(other_table.data[j][col])
                    new_table.append(new_row)
        return MyPyTable(new_header, new_table)

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        self_key_column_indices = [self.column_names.index(key) for key in key_column_names]
        other_key_column_indices = [other_table.column_names.index(key) for key in key_column_names]
        new_header = copy.deepcopy(self.column_names)
        other_non_key_column_names = [name for name in other_table.column_names if name not in key_column_names]
        new_header += other_non_key_column_names
        new_table = []
        for i in range(len(self.data)):
            match_found = False
            for j in range(len(other_table.data)):
                keys_match = True
                for key in range(len(self_key_column_indices)):
                    if self.data[i][self_key_column_indices[key]] != other_table.data[j][other_key_column_indices[key]]:
                        keys_match = False
                if keys_match:
                    match_found = True
                    new_row = []
                    for col in range(len(self.data[i])):
                        new_row.append(self.data[i][col])
                    for col in range(len(other_table.data[j])):
                        if col not in other_key_column_indices:
                            new_row.append(other_table.data[j][col])
                    new_table.append(new_row)
            if not match_found:
                new_row = []
                for col in range(len(self.data[i])):
                    new_row.append(self.data[i][col])
                for col in range(len(other_table.column_names)):
                    if col not in other_key_column_indices:
                        new_row.append("NA")
                new_table.append(new_row)
        for j in range(len(other_table.data)):
            match_found = False
            for i in range(len(self.data)):
                keys_match = True
                for key in range(len(self_key_column_indices)):
                    if self.data[i][self_key_column_indices[key]] != other_table.data[j][other_key_column_indices[key]]:
                        keys_match = False
                if keys_match:
                    match_found = True
            if not match_found:
                new_row = []
                for col in range(len(self.column_names)):
                    if col not in self_key_column_indices:
                        new_row.append("NA")
                    else:
                        col_name = self.column_names[col]
                        col_index = other_table.column_names.index(col_name)
                        new_row.append(other_table.data[j][col_index])
                for col in range(len(other_table.data[j])):
                    if col not in other_key_column_indices:
                        new_row.append(other_table.data[j][col])
                new_table.append(new_row)
        return MyPyTable(new_header, new_table)

header = ["id", "a", "b", "c"]
data = [["ID001", "1", "-5.5", 1.7],
            ["ID002", "A", "2.2", 1.0]]
data_as_numeric = [["ID001", 1, -5.5, 1.7],
            ["ID002", "A", 2.2, 1.0]]

header_dups = ["id", "a", "b"]
data_dups = [["ID001", "A", 1.0],
            ["ID002", "B", 1.5],
            ["ID003", "A", 1.0],
            ["ID002", "A", 1.5],
            ["ID004", "C", 1.0],
            ["ID005", "C", 1.0],
            ["ID006", "D", 1.0],
            ["ID007", "A", 2.0],
            ["ID008", "C", 1.0]]

data_dups_dropped = [["ID001", "A", 1.0],
            ["ID002", "B", 1.5],
            ["ID002", "A", 1.5],
            ["ID004", "C", 1.0],
            ["ID006", "D", 1.0],
            ["ID007", "A", 2.0]]

header_stats = ["a", "b", "c"]
data_stats = [[1.0, 2.0, 3.0],
                [2.5, 2.0, 1.0],
                [0.0, -1.0, 1.0],
                [-2.0, 0.5, 0.0]]

# first trace example (single key)
# adapted from SQL examples at https://www.diffen.com/difference/Inner_Join_vs_Outer_Join
header_left = ["Product", "Price"]
data_left = [["Potatoes", 3.0],
                ["Avacodos", 4.0],
                ["Kiwis", 2.0],
                ["Onions", 1.0],
                ["Melons", 5.0],
                ["Oranges", 5.0],
                ["Tomatoes", 6.0]]
header_right = ["Product", "Quantity"]
data_right = [["Potatoes", 45.0],
                ["Avacodos", 63.0],
                ["Kiwis", 19.0],
                ["Onions", 20.0],
                ["Melons", 66.0],
                ["Broccoli", 27.0],
                ["Squash", 92.0]]

header_NAs = ["id", "a", "b"]
data_NAs = [["ID001", "A", 3.5],
            ["ID002", "B", "NA"],
            ["ID003", "C", 1.0],
            ["ID004", "D", 1.5]]


# second trace example (multiple attribute key)
header_car_left = ["SaleId","EmployeeId","CarName","ModelYear","Amt"]
data_car_left = [[555.0,12.0,"ford pinto",75.0,3076.0],
                [556.0,12.0,"toyota corolla",75.0,2611.0],
                [998.0,13.0,"toyota corolla",75.0,2800.0],
                [999.0,12.0,"toyota corolla",76.0,2989.0]]
header_car_right = ["CarName","ModelYear","MSRP"]
data_car_right = [["ford pinto",75.0,2769.0],
                    ["toyota corolla",75.0,2711.0],
                    ["ford pinto",76.0,3025.0],
                    ["toyota corolla",77.0,2789.0]]

# join practice problem
header_car_left_long = ["SaleId","EmployeeId","CarName","ModelYear","Amt"]
data_car_left_long = [[555.0,12.0,"ford pinto",75.0,3076.0], # full match
    [556.0,12.0,"toyota truck",79.0,2989.0], # 0 match
    [557.0,12.0,"toyota corolla",75.0,2611.0], # full match
    [996.0,13.0,"toyota corolla",75.0,2800.0], # 2nd full match
    [997.0,12.0,"toyota corolla",76.0,2989.0], # match on CarName, match on ModelYear; not together
    [998.0,12.0,"ford pinto",74.0,2989.0], # match on CarName, 0 match on ModelYear
    [999.0,12.0,"ford mustang",77.0,2989.0]] # 0 match on CarName, match on ModelYear
header_car_right_long = ["CarName","ModelYear","MSRP"]
data_car_right_long = [["honda accord",75.0,2789.0], # 0 match on CarName, match on ModelYear
    ["ford pinto",75.0,2769.0], # full match
    ["toyota corolla",75.0,2711.0], # full match
    ["ford pinto",76.0,3025.0], # match on CarName, match on ModelYear; not together
    ["toyota corolla",77.0,2789.0], # match on CarName, match on ModelYear; not together
    ["range rover",70.0,3333.0], # 0 match
    ["ford pinto",73.0,2567.0], # match on CarName, 0 match on ModelYear
    ["toyota corolla",75.0,2999.0]] # 2nd full match
def check_same_lists_regardless_of_order(list1, list2):
    """Utility function
    """
    assert len(list1) == len(list2) # same length
    for item in list1:
        assert item in list2
        list2.remove(item)
    assert len(list2) == 0
    return True
if __name__ == "__main__":
        # single attribute key
    table_left = MyPyTable(header_left, data_left)
    table_right = MyPyTable(header_right, data_right)
    joined_table = table_left.perform_full_outer_join(table_right, ["Product"])
    assert len(joined_table.column_names) == 3
    # test against pandas' outer join
    #df_left = pd.DataFrame(data_left, columns=header_left)
    #df_right = pd.DataFrame(data_right, columns=header_right)
    #df_joined = df_left.merge(df_right, how="outer", on=["Product"])
    #df_joined.fillna("NA", inplace=True)
    #check_same_lists_regardless_of_order(joined_table.data, df_joined.values.tolist())

    # multiple attribute key example from class
    table_left = MyPyTable(header_car_left, data_car_left)
    table_right = MyPyTable(header_car_right, data_car_right)
    joined_table = table_left.perform_full_outer_join(table_right, ["CarName", "ModelYear"])
    assert len(joined_table.column_names) == 6
    # test against pandas' inner join
    #df_left = pd.DataFrame(data_car_left, columns=header_car_left)
    #df_right = pd.DataFrame(data_car_right, columns=header_car_right)
    #df_joined = df_left.merge(df_right, how="outer", on=["CarName", "ModelYear"])
    #df_joined.fillna("NA", inplace=True)
    #check_same_lists_regardless_of_order(joined_table.data, df_joined.values.tolist())

    # join practice problem
    table_left = MyPyTable(header_car_left_long, data_car_left_long)
    table_right = MyPyTable(header_car_right_long, data_car_right_long)
    joined_table = table_left.perform_full_outer_join(table_right, ["CarName", "ModelYear"])
    assert len(joined_table.column_names) == 6
    # test against pandas' outer join
    #df_left = pd.DataFrame(data_car_left_long, columns=header_car_left_long)
    #df_right = pd.DataFrame(data_car_right_long, columns=header_car_right_long)
    #df_joined = df_left.merge(df_right, how="outer", on=["CarName", "ModelYear"])
    #df_joined.fillna("NA", inplace=True)
    #check_same_lists_regardless_of_order(joined_table.data, df_joined.values.tolist())

    # now check non-adjacent composite key columns
    # data prep
    header_car_left_copy = copy.deepcopy(header_car_left_long)
    data_car_left_copy = copy.deepcopy(data_car_left_long)
    header_car_right_copy = copy.deepcopy(header_car_right_long)
    data_car_right_copy = copy.deepcopy(data_car_right_long)
    # swap CarName and SaleId columns, then CarName and ModelYear columns
    # so they are in diff order, 2 apart
    header_car_left_copy[0], header_car_left_copy[2] = \
        header_car_left_copy[2], header_car_left_copy[0]
    header_car_left_copy[0], header_car_left_copy[3] = \
        header_car_left_copy[3], header_car_left_copy[0]
    for row in data_car_left_copy:
        row[0], row[2] = row[2], row[0]
        row[0], row[3] = row[3], row[0]
    # swap ModelYear and MSRP columns so they are 1 apart
    header_car_right_copy[1], header_car_right_copy[2] = \
        header_car_right_copy[2], header_car_right_copy[1]
    for row in data_car_right_copy:
        row[1], row[2] = row[2], row[1]
    # test setup
    table_left = MyPyTable(header_car_left_copy, data_car_left_copy)
    table_right = MyPyTable(header_car_right_copy, data_car_right_copy)
    joined_table = table_left.perform_full_outer_join(table_right, ["CarName", "ModelYear"])
    assert len(joined_table.column_names) == 6
    # test against pandas' outer join
    #df_left = pd.DataFrame(data_car_left_copy, columns=header_car_left_copy)
    #df_right = pd.DataFrame(data_car_right_copy, columns=header_car_right_copy)
    #df_joined = df_left.merge(df_right, how="outer", on=["CarName", "ModelYear"])
    #df_joined.fillna("NA", inplace=True)
    #check_same_lists_regardless_of_order(joined_table.data, df_joined.values.tolist())

