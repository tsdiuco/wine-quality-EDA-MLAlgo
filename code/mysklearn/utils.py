import copy
import math

def get_column(table, header, col_name):
    """Extracts a column from the table data as a list.
    Args:
        table(MyPyTable): Data from MyPyTable
        header(MyPyTable): Column headers associated with table
        col_name(string): Column to be identified
    Returns:
        list of obj: 1D list of values in the column
    """
    col_index = header.index(col_name)
    col = []
    for row in table:
        value = row[col_index]
        if value != "N/A":
            col.append(value)
    return col

def get_frequencies(table, header, col_name):
    """Get the frequency of each value
    Args:
        table(MyPyTable): Data from MyPyTable
        header(MyPyTable): Column headers associated with table
        col_name(string): Column to be identified
    Returns:
        values: different values
        count: count of each value
    """
    column = get_column(table, header, col_name)
    column.sort() # inplace sort
    values = []
    counts = []
    for value in column:
        if value in values:
            counts[-1] += 1 # okay because col is sorted
        else: # haven't seen this value before
            values.append(value)
            counts.append(1)
    return values, counts # we can return multiple items

def sum_column(table, header, col_name):
    """Sum the contents of a column
    Args:
        table(MyPyTable): Data from MyPyTable
        header(MyPyTable): Column headers associated with table
        col_name(string): Column to be identified
    Returns:
        total = total of the column
    """
    column = get_column(table, header, col_name)
    total = 0

    for row in column:
        total += row

    return total

def categorize_by_mpg_rating(data_table):
    """Specifically for mpg ratings categorizes by rating
    Args:
        data_table(list): List of cars and the mpg
    Returns:
        ratings: Ratings categories
        count_ratings: Number in each category
    """
    table_copy = copy.deepcopy(data_table)
    mpg_column = get_column(table_copy.data, table_copy.column_names, "mpg")

    # Parallel lists to keep track of frequencies
    rating =        [1,2,3,4,5,6,7,8,9,10]
    count_ratings = [0,0,0,0,0,0,0,0,0,0]

    for row in mpg_column:
        if row >= 45:
            count_ratings[9] += 1
        if row >= 37:
            count_ratings[8] += 1
        if row >= 31:
            count_ratings[7] += 1
        if row >= 27:
            count_ratings[6] += 1
        if row >= 24:
            count_ratings[5] += 1
        if row >= 20:
            count_ratings[4] += 1
        if row >= 17:
            count_ratings[3] += 1
        if row >= 15:
            count_ratings[2] += 1
        if row >= 14:
            count_ratings[1] += 1
        else:
            count_ratings[0] += 1

    return rating, count_ratings

def compute_equal_width_cutoffs(values, num_bins):
    """Determines cutoff points to be used in equal bin width categorzation
    Args:
        values(list): values to be put in bins
        num_bins(int): Number of bins to create
    Returns:
        cutoffs(list): list of bin cutoff points
    """
    values_range = max(values) - min(values)
    bin_width = values_range / num_bins
    start = min(values)
    cutoffs = []

    hold = 0

    for i in range(num_bins):
        cutoff = start + bin_width
        cutoffs.append(cutoff)
        start = cutoff
        hold = hold + i

    cutoffs.insert(0, min(values))

    return [round(cutoff, 2) for cutoff in cutoffs]

def compute_bin_frequencies(values, cutoffs):
    """Categorizes values in to bins
    Args:
        values(list): values to be put in bins
        cutoffs(list): list of ranges of each bin
    Returns:
        freqs(list): number in each bin
    """
    freqs = [0 for _ in range(len(cutoffs) - 1)]

    for value in values:
        if value == max(values):
            freqs[-1] += 1 # increment the last bin's freq
        else:
            for i in range(len(cutoffs) - 1):
                if cutoffs[i] <= value < cutoffs[i + 1]:
                    freqs[i] += 1
    return freqs

def compute_covariance_and_coefficient(xdata, ydata):
    """Computes the covariance and coefficient of data
    Args:
        xdata(list): set of data
        ydata(list): set of data
    Returns:
        coefficient(float)
        covariance(float)
    """
    xmean = sum(xdata) / len(xdata)
    ymean = sum(ydata) / len(ydata)

    summation = 0

    table_range = range(len(xdata))

    for i in table_range:
        summation += (xdata[i]-xmean) *(ydata[i]-ymean)
    coefficient_denomenator = math.sqrt((sum([(xdata[i] - xmean) ** 2 \
        for i in range(len(xdata))]) * sum([(ydata[i] - ymean) ** 2 for i in range(len(ydata))])))
    covariance = summation/len(xdata)
    coefficient = summation/coefficient_denomenator

    return coefficient, covariance

def convert_to_numeric(column):
    """Try to convert each value in the table to a numeric type (float).
    Notes:
        Leave values as is that cannot be converted to numeric.
    """

    # Traverses each row of data
    list_1 = []
    for row in column:
    # for row in range(len(self.data)):
    # Traverses each coloumn in a row
        try:
            list_1.append(float(row))
        # Because we are attempting to convert to numberic for every item in the 2d list
        # if the value is not convertable, we just pass the error and assume it is there
        except ValueError:
            pass
            #print(self.data[row][col], " could not be converted to numeric")
    return list_1

def remove_percentage(column):
    """Removes percentage sign if exists
    Returns: New list without %
    """
    list_0 = []
    for row in column:
        list_0.append(row.replace('%', ''))
    return list_0