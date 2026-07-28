import pandas as pd

def get_class_counts(file_path, column_name='attribute'):
    """
    Reads an Excel file and returns case-insensitive counts of classes in a specified column.
    """
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)
        
        # Map column names to lowercase to handle 'Attribute', 'ATTRIBUTE', etc. safely
        col_map = {str(col).lower(): col for col in df.columns}
        target_col_lower = column_name.lower()
        
        if target_col_lower not in col_map:
            print(f"Available columns in the file: {list(df.columns)}")
            raise ValueError(f"Error: The column '{column_name}' was not found.")
            
        # Use the actual column name found in the file
        actual_col_name = col_map[target_col_lower]
            
        # Convert values to lowercase strings and count them
        counts = df[actual_col_name].astype(str).str.lower().value_counts()
        
        return counts
        
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Example Usage ---
# Replace 'data.csv' with the actual path to your file
file_path = r"C:\Users\naray\Downloads\May June July Image Extraction.xlsx" 
class_counts = get_class_counts(file_path, column_name='attribute')

if class_counts is not None:
    print("Class Counts (Case-Insensitive):")
    print("-" * 35)
    print(class_counts)