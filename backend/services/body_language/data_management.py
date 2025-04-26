import os
import csv
import pandas as pd
from typing import List

class BodyLanguageDataManager:
    """Handles data management operations like file access and data cleanup."""
    
    def __init__(self, service):
        self.service = service
    
    def get_available_classes(self) -> List[str]:
        """Get list of available classes in the training data"""
        if not os.path.exists(self.service.data_path) or os.path.getsize(self.service.data_path) == 0:
            return []
            
        df = pd.read_csv(self.service.data_path)
        return df.iloc[:, 0].unique().tolist()
    
    def delete_all_data(self) -> bool:
        """Delete all training data, models, and feature count information"""
        try:
            # Delete data file
            if os.path.exists(self.service.data_path):
                os.remove(self.service.data_path)
                # Create empty file
                with open(self.service.data_path, 'w', newline='') as f:
                    pass
            
            # Delete model file
            if os.path.exists(self.service.model_path):
                os.remove(self.service.model_path)
            
            # Delete feature count file
            if os.path.exists(self.service.feature_count_path):
                os.remove(self.service.feature_count_path)
            
            # Reset model
            self.service.model = None
            
            return True
        except Exception as e:
            print(f"Error deleting training data: {str(e)}")
            return False
    
    def repair_data_file(self) -> bool:
        """Repair the data file by ensuring all rows have the same number of columns"""
        if not os.path.exists(self.service.data_path) or os.path.getsize(self.service.data_path) == 0:
            print("No data file to repair.")
            return False
        
        try:
            # Find max columns in the file
            max_cols = 0
            num_rows = 0
            class_names = []
            
            with open(self.service.data_path, 'r', newline='') as f:
                csv_reader = csv.reader(f)
                header = next(csv_reader)  # Read header
                max_cols = len(header)
                
                for i, row in enumerate(csv_reader):
                    num_rows += 1
                    class_names.append(row[0])  # Save class name
                    cols = len(row)
                    if cols > max_cols:
                        max_cols = cols
            
            if max_cols == len(header):
                print("Data file appears to be consistent.")
                return True
            
            print(f"Repairing data file with {max_cols} columns")
            
            # Create a new file with consistent columns
            temp_file = self.service.data_path + ".temp"
            with open(temp_file, 'w', newline='') as f_new:
                csv_writer = csv.writer(f_new)
                
                # Create new header
                new_header = ['class'] + [f'feature_{i}' for i in range(1, max_cols)]
                csv_writer.writerow(new_header)
                
                # Reread and pad rows
                with open(self.service.data_path, 'r', newline='') as f:
                    csv_reader = csv.reader(f)
                    next(csv_reader)  # Skip header
                    
                    for i, row in enumerate(csv_reader):
                        # Pad row with zeros if needed
                        padded_row = row + ['0'] * (max_cols - len(row))
                        csv_writer.writerow(padded_row)
            
            # Replace original file
            os.replace(temp_file, self.service.data_path)
            
            # Update feature count file
            with open(self.service.feature_count_path, 'w') as f:
                f.write(str(max_cols - 1))  # -1 for class column
                
            print(f"Successfully repaired data file")
            return True
            
        except Exception as e:
            print(f"Error repairing data file: {str(e)}")
            return False 