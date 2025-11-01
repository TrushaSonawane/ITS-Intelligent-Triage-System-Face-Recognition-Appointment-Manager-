import cv2
import os
import pickle
import numpy as np
import face_recognition
from datetime import datetime
import json
import tkinter as tk
from tkinter import messagebox, font as tkfont, scrolledtext
from functools import partial
import calendar
from time import strftime

# ===============================
# CONFIGURATION & CONSTANTS
# ===============================
KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE = "encodings.pkl"
PATIENT_DATA_FILE = "patient_data.json" # Stores permanent patient records
APPOINTMENTS_FILE = "appointments.json" # Stores current appointments
DOCTOR_DATA_FILE = "doctor_data.json" # Stores doctor records
TOLERANCE = 0.6
FRAME_RESIZE_SCALE = 0.25  # 1/4 size for faster processing
MODEL = "hog"
FONT = cv2.FONT_HERSHEY_DUPLEX
NUM_REGISTRATION_IMAGES = 5 # Images to capture for new user
INFO_PANEL_WIDTH = 400 # Width of the new data panel in the CV2 window
# NEW: Define a standardized size for Tkinter windows (e.g., 600x800)
TK_WINDOW_GEOMETRY = "600x800" 

# Color codes for terminal logging (ANSI)
CYAN = '\033[96m'
NEON_GREEN = '\033[92m'
CRITICAL_RED = '\033[91m'
YELLOW = '\033[93m'
BOLD = '\033[1m'
END = '\033[0m'

# Global variables for real-time data sharing
PATIENT_RECORDS = {} 
APPOINTMENTS = []
DOCTOR_RECORDS = {} # Global variable for doctor records
KNOWN_ENCODINGS = []
KNOWN_NAMES = []

# ===============================
# 1. STYLIZED LOGGING FUNCTION
# ===============================

def cyber_log(message, level="INFO"):
    """Prints a stylized message to the console."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    if level == "INFO":
        color = CYAN
        prefix = f"[{BOLD}SYS-LOG{END}]"
    elif level == "WARNING":
        color = YELLOW
        prefix = f"[{BOLD}WARNING{END}]"
    elif level == "CRITICAL":
        color = CRITICAL_RED
        prefix = f"[{BOLD}ALERT-TRIAGE{END}]"
    elif level == "SUCCESS":
        color = NEON_GREEN
        prefix = f"[{BOLD}STATUS-OK{END}]"
    else:
        color = ""
        prefix = "[LOG]"

    print(f"{color}{prefix} <{timestamp}> {message}{END}")

# ===============================
# 2. JSON & ENCODING HANDLERS (Patient Data)
# ===============================

def load_patient_records():
    """Loads patient records from the JSON file into the global dictionary."""
    global PATIENT_RECORDS
    if os.path.exists(PATIENT_DATA_FILE):
        try:
            with open(PATIENT_DATA_FILE, 'r') as f:
                PATIENT_RECORDS = json.load(f)
            cyber_log(f"Loaded {len(PATIENT_RECORDS)} patient records from {PATIENT_DATA_FILE}", "INFO")
        except json.JSONDecodeError:
            cyber_log("ERROR: Could not decode JSON data. Starting with empty records.", "CRITICAL")
            PATIENT_RECORDS = {}
    else:
        cyber_log(f"{PATIENT_DATA_FILE} not found. Creating empty file.", "WARNING")
        PATIENT_RECORDS = {}
        save_patient_records_to_file()

def save_patient_records_to_file():
    """Saves the current global patient records to the JSON file."""
    with open(PATIENT_DATA_FILE, 'w') as f:
        json.dump(PATIENT_RECORDS, f, indent=4)

def save_patient_record(name, data):
    """Saves a single patient record to the global dictionary and JSON file."""
    global PATIENT_RECORDS
    PATIENT_RECORDS[name] = data
    save_patient_records_to_file()
    cyber_log(f"Patient {name} data saved to JSON.", "SUCCESS")

# ===============================
# 2b. APPOINTMENT HANDLERS
# ===============================

def load_appointments():
    """Loads appointments from the JSON file into the global list."""
    global APPOINTMENTS
    if os.path.exists(APPOINTMENTS_FILE):
        try:
            with open(APPOINTMENTS_FILE, 'r') as f:
                APPOINTMENTS = json.load(f)
            # Sort appointments by date and time for better viewing
            APPOINTMENTS.sort(key=lambda x: (x['date'], x['time']))
            cyber_log(f"Loaded {len(APPOINTMENTS)} pending appointments from {APPOINTMENTS_FILE}", "INFO")
        except json.JSONDecodeError:
            cyber_log("ERROR: Could not decode appointments JSON data. Starting with empty list.", "CRITICAL")
            APPOINTMENTS = []
    else:
        cyber_log(f"{APPOINTMENTS_FILE} not found. Creating empty file.", "WARNING")
        APPOINTMENTS = []
        save_appointments()

def save_appointments():
    """Saves the current global appointments list to the JSON file."""
    with open(APPOINTMENTS_FILE, 'w') as f:
        json.dump(APPOINTMENTS, f, indent=4)
        
def add_appointment(data):
    """Adds a new appointment to the global list and saves."""
    global APPOINTMENTS
    APPOINTMENTS.append(data)
    # Re-sort list after adding new appointment
    APPOINTMENTS.sort(key=lambda x: (x['date'], x['time'])) 
    save_appointments()
    cyber_log(f"New appointment booked for {data['name']} on {data['date']} at {data['time']} with Dr. {data['doctor']}.", "SUCCESS")


# ===============================
# 2c. ENCODING HANDLERS
# ===============================

def load_known_faces():
    """Load known faces from saved pickle file or rebuild from images."""
    if os.path.exists(ENCODINGS_FILE):
        cyber_log("Loading known face encodings from file...", "INFO")
        try:
            with open(ENCODINGS_FILE, "rb") as f:
                data = pickle.load(f)
            return data["encodings"], data["names"]
        except Exception as e:
            cyber_log(f"ERROR: Could not load encodings file: {e}. Starting fresh.", "CRITICAL")
            return [], []

    cyber_log("Encodings file not found. Starting with empty encodings.", "WARNING")
    return [], []

def save_encodings(encodings, names):
    """Saves the current list of encodings and names to a pickle file."""
    data = {"encodings": encodings, "names": names}
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)
    cyber_log(f"Saved {len(names)} face encodings to {ENCODINGS_FILE}.", "SUCCESS")

# ===============================
# 2d. DOCTOR HANDLERS
# ===============================

def load_doctor_records():
    """Loads doctor records from the JSON file into the global dictionary."""
    global DOCTOR_RECORDS
    if os.path.exists(DOCTOR_DATA_FILE):
        try:
            with open(DOCTOR_DATA_FILE, 'r') as f:
                DOCTOR_RECORDS = json.load(f)
            cyber_log(f"Loaded {len(DOCTOR_RECORDS)} doctor records from {DOCTOR_DATA_FILE}", "INFO")
        except json.JSONDecodeError:
            cyber_log("ERROR: Could not decode Doctor JSON data. Starting with empty records.", "CRITICAL")
            DOCTOR_RECORDS = {}
    else:
        cyber_log(f"{DOCTOR_DATA_FILE} not found. Creating empty file.", "WARNING")
        DOCTOR_RECORDS = {}
        save_doctor_records()

def save_doctor_records():
    """Saves the current global doctor records to the JSON file."""
    with open(DOCTOR_DATA_FILE, 'w') as f:
        json.dump(DOCTOR_RECORDS, f, indent=4)

def save_doctor_record(name, data):
    """Saves a single doctor record to the global dictionary and JSON file."""
    global DOCTOR_RECORDS
    DOCTOR_RECORDS[name] = data
    save_doctor_records()
    cyber_log(f"Doctor {name} data saved to JSON.", "SUCCESS")


# ===============================
# 3. TKINTER FORMS & MENUS 
# ===============================

def create_registration_form():
    """Creates a Tkinter form to collect new patient data."""
    root = tk.Tk()
    root.title("Patient Registration Form (DATA INPUT)")
    root.geometry("600x400") # Smaller geometry for this form
    
    # Custom styling
    root.configure(bg="#0D1117") # Dark background
    font_style = tkfont.Font(family="Consolas", size=10)
    
    form_data = {}

    fields = [("Name (ID)", "name"), ("Age", "age"), ("Gender", "gender"), 
              ("Allergies (or None)", "allergies"), ("History (CRITICAL: if needed)", "history")]
    entries = {}

    for i, (label_text, key) in enumerate(fields):
        # Labels
        tk.Label(root, text=f"{label_text}:", bg="#0D1117", fg="#00FFCC", font=font_style).grid(row=i, column=0, padx=10, pady=5, sticky="w")
        # Entries
        entry = tk.Entry(root, width=40, bg="#21262D", fg="#00FFCC", insertbackground="#00FFCC", bd=1, relief="solid", font=font_style)
        entry.grid(row=i, column=1, padx=10, pady=5)
        entries[key] = entry

    def submit():
        data = {key: entry.get().strip() for key, entry in entries.items()}
        
        if not all(data.values()):
            messagebox.showerror("Error", "All fields must be filled.")
            return

        name = data["name"]
        if name.lower() == "unknown" or not name:
             messagebox.showerror("Error", "Name cannot be 'Unknown' or empty.")
             return

        if name in PATIENT_RECORDS:
             messagebox.showerror("Error", f"Patient ID '{name}' already exists. Use the Existing User flow to update records.")
             return

        form_data["name"] = name
        form_data["details"] = {
            "age": data["age"],
            "gender": data["gender"],
            "allergies": data["allergies"],
            "history": data["history"],
            "last_visit": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        root.destroy()

    # Button
    button_style = {'bg': '#00FFCC', 'fg': '#0D1117', 'activebackground': '#00B38F', 'activeforeground': '#0D1117', 'bd': 0, 'font': font_style}
    tk.Button(root, text="Register & Capture Face Data", command=submit, **button_style).grid(row=len(fields), column=0, columnspan=2, pady=15, padx=10, sticky="ew")
    
    root.protocol("WM_DELETE_WINDOW", lambda: (root.destroy(), form_data.clear())) # Clear data if window is closed
    root.mainloop() 
    return form_data
    

def create_appointment_form():
    """
    Creates a Tkinter form to book a new appointment using dynamic drop-downs for date (YYYY, Month Name, Day) 
    and a 12-hour clock (H, M, AM/PM). Also includes Doctor selection.
    """
    global DOCTOR_RECORDS
    root = tk.Tk()
    root.title("Appointment Booking System")
    root.geometry("600x400") # Smaller geometry
    
    root.configure(bg="#0D1117") 
    font_style = tkfont.Font(family="Consolas", size=10)
    
    # Global variables for drop-down management
    year_var = tk.StringVar(root)
    month_var = tk.StringVar(root)
    day_var = tk.StringVar(root)
    hour_var = tk.StringVar(root)
    minute_var = tk.StringVar(root)
    ampm_var = tk.StringVar(root)
    doctor_var = tk.StringVar(root) # Doctor selection

    # Helper to create styled OptionMenus
    def create_option_menu(parent, variable, values, default_value, width=5):
        variable.set(default_value)
        menu = tk.OptionMenu(parent, variable, *values)
        menu.config(width=width, bg="#21262D", fg="#00FFCC", bd=1, relief="solid", highlightthickness=0, 
                   activebackground="#00B38F", activeforeground="#0D1117", font=font_style)
        menu["menu"].config(bg="#21262D", fg="#00FFCC", font=font_style)
        return menu

    # --- 1. Patient Name and Reason ---
    tk.Label(root, text="Patient Name/ID:", bg="#0D1117", fg="#00FFCC", font=font_style).grid(row=0, column=0, padx=10, pady=5, sticky="w")
    patient_entry = tk.Entry(root, width=40, bg="#21262D", fg="#00FFCC", insertbackground="#00FFCC", bd=1, relief="solid", font=font_style)
    patient_entry.grid(row=0, column=1, columnspan=3, padx=10, pady=5, sticky="ew")

    tk.Label(root, text="Appointment Type/Reason:", bg="#0D1117", fg="#00FFCC", font=font_style).grid(row=1, column=0, padx=10, pady=5, sticky="w")
    reason_entry = tk.Entry(root, width=40, bg="#21262D", fg="#00FFCC", insertbackground="#00FFCC", bd=1, relief="solid", font=font_style)
    reason_entry.grid(row=1, column=1, columnspan=3, padx=10, pady=5, sticky="ew")

    # --- 2. Date Selection (Year, Month, Day) ---
    tk.Label(root, text="Date:", bg="#0D1117", fg="#00FFCC", font=font_style).grid(row=2, column=0, padx=10, pady=5, sticky="w")
    
    # Date Frame
    date_frame = tk.Frame(root, bg="#0D1117")
    date_frame.grid(row=2, column=1, columnspan=3, padx=10, pady=5, sticky="w")
    
    # Year Dropdown (Current year + 10 years)
    current_year = datetime.now().year
    years = [str(y) for y in range(current_year, current_year + 11)]
    year_menu = create_option_menu(date_frame, year_var, years, str(current_year), width=6)
    year_menu.pack(side=tk.LEFT)
    
    # Month Dropdown (Month Names)
    month_names = list(calendar.month_abbr)[1:] # Jan, Feb, Mar...
    month_menu = create_option_menu(date_frame, month_var, month_names, calendar.month_abbr[datetime.now().month], width=6)
    month_menu.pack(side=tk.LEFT, padx=(5, 5))

    # Day Dropdown (Dynamic)
    day_menu = tk.OptionMenu(date_frame, day_var, "") # Placeholder
    day_menu.config(width=4, bg="#21262D", fg="#00FFCC", bd=1, relief="solid", highlightthickness=0, 
                   activebackground="#00B38F", activeforeground="#0D1117", font=font_style)
    day_menu["menu"].config(bg="#21262D", fg="#00FFCC", font=font_style)
    day_menu.pack(side=tk.LEFT)
    
    def update_days(*args):
        try:
            selected_year = int(year_var.get())
            selected_month_abbr = month_var.get()
            selected_month = month_names.index(selected_month_abbr) + 1
            
            # Get number of days in the selected month/year
            _, num_days = calendar.monthrange(selected_year, selected_month)
            days = [str(d).zfill(2) for d in range(1, num_days + 1)]
            
            # Preserve selected day if possible, otherwise reset
            current_day = day_var.get()
            if current_day not in days:
                day_var.set(days[0]) # Default to the 1st
                
            # Update the OptionMenu with the new list of days
            day_menu['menu'].delete(0, 'end')
            for day in days:
                day_menu['menu'].add_command(label=day, command=tk._setit(day_var, day))
                
        except (ValueError, IndexError):
            pass

    # Link the update function to changes in Year and Month
    year_var.trace_add("write", update_days)
    month_var.trace_add("write", update_days)
    
    # Initial call to populate the day list
    update_days()
    day_var.set(str(datetime.now().day).zfill(2)) # Default to today's day

    # --- 3. Time Selection (12-Hour Clock) ---
    tk.Label(root, text="Time:", bg="#0D1117", fg="#00FFCC", font=font_style).grid(row=3, column=0, padx=10, pady=5, sticky="w")

    # Time Frame
    time_frame = tk.Frame(root, bg="#0D1117")
    time_frame.grid(row=3, column=1, columnspan=3, padx=10, pady=5, sticky="w")
    
    # Hour Dropdown (1-12)
    hours = [str(h).zfill(2) for h in range(1, 13)]
    hour_menu = create_option_menu(time_frame, hour_var, hours, strftime('%I'), width=4)
    hour_menu.pack(side=tk.LEFT)
    
    tk.Label(time_frame, text=":", bg="#0D1117", fg="#00FFCC", font=font_style).pack(side=tk.LEFT)

    # Minute Dropdown (00, 15, 30, 45)
    minutes = ["00", "15", "30", "45"]
    current_minute = strftime('%M')
    default_minute = min(minutes, key=lambda x: abs(int(x) - int(current_minute)))
    minute_menu = create_option_menu(time_frame, minute_var, minutes, default_minute, width=4)
    minute_menu.pack(side=tk.LEFT)

    # AM/PM Dropdown
    ampm = ["AM", "PM"]
    ampm_menu = create_option_menu(time_frame, ampm_var, ampm, strftime('%p'), width=4)
    ampm_menu.pack(side=tk.LEFT, padx=(10, 0))

    # --- 4. Assign Doctor (Using live DOCTOR_RECORDS) ---
    doctor_names = list(DOCTOR_RECORDS.keys())
    if not doctor_names:
        doctor_names = ["No Doctors Registered"]
        
    doctor_var.set(doctor_names[0]) # Default value
    
    tk.Label(root, text="Assign Doctor:", bg="#0D1117", fg="#00FFCC", font=font_style).grid(row=4, column=0, padx=10, pady=5, sticky="w")
    
    doctor_menu = tk.OptionMenu(root, doctor_var, *doctor_names)
    doctor_menu.config(width=37, bg="#21262D", fg="#00FFCC", bd=1, relief="solid", highlightthickness=0, 
                        activebackground="#00B38F", activeforeground="#0D1117", font=font_style)
    doctor_menu["menu"].config(bg="#21262D", fg="#00FFCC", font=font_style)
    doctor_menu.grid(row=4, column=1, columnspan=3, padx=10, pady=5, sticky="ew")

    def submit():
        patient_name = patient_entry.get().strip()
        reason = reason_entry.get().strip()
        doctor_assigned = doctor_var.get()
        
        # 1. Basic Validation
        if not patient_name or not reason:
            messagebox.showerror("Error", "Patient Name and Appointment Reason must be filled.")
            return

        if doctor_assigned == "No Doctors Registered":
            messagebox.showerror("Error", "Cannot book appointment. Please register a doctor first via Doctor Management.")
            return
            
        # 2. Convert Date Dropdowns to YYYY-MM-DD and Time to 24h HH:MM
        try:
            year = year_var.get()
            month_index = month_names.index(month_var.get()) + 1
            day = day_var.get()
            
            formatted_date = f"{year}-{str(month_index).zfill(2)}-{day}"
            
            # Convert 12h time to 24h time for internal storage
            time_format_str = f"{hour_var.get()}:{minute_var.get()} {ampm_var.get()}"
            appt_datetime = datetime.strptime(f"{formatted_date} {time_format_str}", "%Y-%m-%d %I:%M %p")
            
            formatted_time = appt_datetime.strftime("%H:%M") # Store as 24h
            
            if appt_datetime < datetime.now():
                messagebox.showerror("Error", "Appointment date/time cannot be in the past.")
                return

        except ValueError:
            messagebox.showerror("Error", "Invalid Date or Time selection. Please ensure all drop-downs are selected.")
            return

        # 3. Save Appointment
        data = {
            "name": patient_name,
            "date": formatted_date,
            "time": formatted_time,
            "reason": reason,
            "doctor": doctor_assigned
        }

        add_appointment(data)
        messagebox.showinfo("Success", f"Appointment booked for {patient_name} with Dr. {doctor_assigned} on {formatted_date} at {formatted_time} (24h).")
        root.destroy()

    button_style = {'bg': '#00FFCC', 'fg': '#0D1117', 'activebackground': '#00B38F', 'activeforeground': '#0D1117', 'bd': 0, 'font': font_style}
    tk.Button(root, text="Confirm Appointment", command=submit, **button_style).grid(row=5, column=0, columnspan=4, pady=15, padx=10, sticky="ew")
    
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop() 

# NEW FUNCTION: MANAGE APPOINTMENTS (View + Cancel)
def manage_appointments_window():
    """Creates a Tkinter window to display all appointments and allow cancellation."""
    global APPOINTMENTS
    root = tk.Tk()
    root.title(f"Appointment Management Console ({len(APPOINTMENTS)} Pending)")
    root.geometry(TK_WINDOW_GEOMETRY) # Apply standardized geometry
    
    # Define Cyberpunk Style
    BG_COLOR = "#0D1117"
    FG_COLOR = "#00FFCC"
    
    root.configure(bg=BG_COLOR)
    title_font = tkfont.Font(family="Consolas", size=14, weight="bold")
    content_font = tkfont.Font(family="Consolas", size=10)

    # Title
    tk.Label(root, text=":: SCHEDULED APPOINTMENTS LOG ::", bg=BG_COLOR, fg=FG_COLOR, font=title_font).pack(pady=10, padx=20)
    
    # Scrolled Text Widget for displaying the list
    # Reduced width to fit the 600px geometry
    text_area = scrolledtext.ScrolledText(
        root,
        wrap=tk.WORD,
        width=60, # Adjusted width
        height=25,
        bg="#21262D",
        fg=FG_COLOR,
        insertbackground=FG_COLOR,
        font=content_font,
        bd=0,
        relief=tk.FLAT,
        padx=10,
        pady=10
    )
    text_area.pack(pady=10, padx=10)
    
    # Format and insert appointment data
    if APPOINTMENTS:
        # Adjusted header length for 60-width display (approx 65 characters)
        header = f"{'ID':<4} | {'PATIENT ID':<15} | {'DATE':<10} | {'TIME':<6} | {'DOCTOR':<10} | REASON\n" 
        separator = "-" * 65 + "\n"
        text_area.insert(tk.END, header, 'header')
        text_area.insert(tk.END, separator)
        
        text_area.tag_config('header', foreground="#FF00CC", font=title_font)
        
        for i, appt in enumerate(APPOINTMENTS):
            # i+1 is the user-facing appointment ID
            doctor_name = appt.get('doctor', 'Unassigned')
            # Truncate reason for single-line display
            reason_display = appt['reason'][:20] + '...' if len(appt['reason']) > 23 else appt['reason']
            
            line = (
                f"{i+1:<4} | "
                f"{appt['name']:<15} | "
                f"{appt['date']:<10} | "
                f"{appt['time']:<6} | "
                f"{doctor_name:<10} | "
                f"{reason_display}\n"
            )
            text_area.insert(tk.END, line)
    else:
        text_area.insert(tk.END, ":: NO CURRENT APPOINTMENTS FOUND ::")
        
    text_area.configure(state='disabled') # Make it read-only
    
    # CANCELLATION SECTION
    cancellation_frame = tk.Frame(root, bg=BG_COLOR)
    cancellation_frame.pack(pady=15)
    
    tk.Label(cancellation_frame, text="CANCEL APPOINTMENT ID:", bg=BG_COLOR, fg="#FF00CC", font=content_font).pack(side=tk.LEFT, padx=(0, 10))
    
    appt_id_entry = tk.Entry(cancellation_frame, width=5, bg="#21262D", fg="#00FFCC", insertbackground="#00FFCC", bd=1, relief="solid", font=content_font)
    appt_id_entry.pack(side=tk.LEFT)
    
    def cancel_appointment():
        try:
            # 1. Get user input (1-based index)
            appt_id = int(appt_id_entry.get().strip())
            # Convert to 0-based list index
            appt_index = appt_id - 1
            
            # 2. Validation
            if appt_index < 0 or appt_index >= len(APPOINTMENTS):
                messagebox.showerror("Error", f"Invalid Appointment ID: {appt_id}. Please enter a valid ID from the list.")
                return

            # 3. Confirmation
            appt_to_cancel = APPOINTMENTS[appt_index]
            confirm = messagebox.askyesno(
                "Confirm Cancellation", 
                f"Are you sure you want to cancel the appointment for:\nPatient: {appt_to_cancel['name']}\nDate: {appt_to_cancel['date']} at {appt_to_cancel['time']}?"
            )
            
            if confirm:
                # 4. Deletion
                del APPOINTMENTS[appt_index]
                save_appointments() # Save changes to the file
                cyber_log(f"Appointment ID {appt_id} for {appt_to_cancel['name']} has been cancelled.", "SUCCESS")
                messagebox.showinfo("Success", f"Appointment ID {appt_id} has been successfully cancelled.")
                
                # 5. Refresh the window
                root.destroy()
                manage_appointments_window()
            
        except ValueError:
            messagebox.showerror("Error", "Please enter a number for the Appointment ID.")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

    button_style = {'bg': '#FF00CC', 'fg': '#0D1117', 'activebackground': '#B3008F', 'activeforeground': '#0D1117', 'bd': 0, 'font': content_font}
    tk.Button(cancellation_frame, text="Cancel Selected", command=cancel_appointment, **button_style).pack(side=tk.LEFT, padx=(15, 0))
    
    # Close button
    close_button_style = {'bg': '#00FFCC', 'fg': '#0D1117', 'activebackground': '#00B38F', 'activeforeground': '#0D1117', 'bd': 0, 'font': content_font, 'width': 15}
    tk.Button(root, text="CLOSE", command=root.destroy, **close_button_style).pack(pady=15)
    
    root.mainloop()

def create_doctor_registration_form():
    """
    Creates a Tkinter form to collect new doctor data with segmented scheduling (Days and Times).
    """
    root = tk.Tk()
    root.title("Doctor Registration Form (DATA INPUT)")
    root.geometry("600x450") # Smaller geometry
    
    root.configure(bg="#0D1117")
    font_style = tkfont.Font(family="Consolas", size=10)
    
    entries = {}
    row_idx = 0

    # 1. Basic Fields
    fields = [("Name (ID)", "name"), ("Specialization", "specialization"), ("Contact", "contact")]
    for i, (label_text, key) in enumerate(fields):
        tk.Label(root, text=f"{label_text}:", bg="#0D1117", fg="#00FFCC", font=font_style).grid(row=i, column=0, padx=10, pady=5, sticky="w")
        entry = tk.Entry(root, width=40, bg="#21262D", fg="#00FFCC", insertbackground="#00FFCC", bd=1, relief="solid", font=font_style)
        entry.grid(row=i, column=1, padx=10, pady=5)
        entries[key] = entry
        row_idx = i
    
    row_idx += 1
    
    # 2. Available Days (Checkboxes)
    tk.Label(root, text="Available Days:", bg="#0D1117", fg="#00FFCC", font=font_style).grid(row=row_idx, column=0, padx=10, pady=5, sticky="nw")
    days_frame = tk.Frame(root, bg="#0D1117")
    days_frame.grid(row=row_idx, column=1, padx=10, pady=5, sticky="w")
    row_idx += 1
    
    days_of_week = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    day_vars = {}
    
    for i, day in enumerate(days_of_week):
        var = tk.IntVar(value=0) # 0 for unchecked, 1 for checked
        chk = tk.Checkbutton(days_frame, text=day, variable=var, 
                             bg="#0D1117", fg="#00FFCC", selectcolor="#21262D", 
                             activebackground="#0D1117", activeforeground="#00FFCC", 
                             font=font_style)
        chk.grid(row=i // 4, column=i % 4, sticky="w")
        day_vars[day] = var
        
    # 3. Available Timings
    tk.Label(root, text="Available Time (HH:MM):", bg="#0D1117", fg="#00FFCC", font=font_style).grid(row=row_idx, column=0, padx=10, pady=5, sticky="w")
    time_frame = tk.Frame(root, bg="#0D1117")
    time_frame.grid(row=row_idx, column=1, padx=10, pady=5, sticky="w")
    row_idx += 1
    
    tk.Label(time_frame, text="From:", bg="#0D1117", fg="#00FFCC", font=font_style).pack(side=tk.LEFT, padx=(0, 5))
    start_time_entry = tk.Entry(time_frame, width=10, bg="#21262D", fg="#00FFCC", insertbackground="#00FFCC", bd=1, relief="solid", font=font_style)
    start_time_entry.pack(side=tk.LEFT, padx=(0, 15))
    entries["start_time"] = start_time_entry
    
    tk.Label(time_frame, text="To:", bg="#0D1117", fg="#00FFCC", font=font_style).pack(side=tk.LEFT, padx=(0, 5))
    end_time_entry = tk.Entry(time_frame, width=10, bg="#21262D", fg="#00FFCC", insertbackground="#00FFCC", bd=1, relief="solid", font=font_style)
    end_time_entry.pack(side=tk.LEFT)
    entries["end_time"] = end_time_entry

    def validate_time(time_str):
        try:
            datetime.strptime(time_str, "%H:%M")
            return True
        except ValueError:
            return False

    def submit():
        data = {key: entry.get().strip() for key, entry in entries.items() if key not in ["start_time", "end_time"]}
        
        # Collect schedule data
        available_days = [day for day, var in day_vars.items() if var.get() == 1]
        start_time = entries["start_time"].get().strip()
        end_time = entries["end_time"].get().strip()

        if not all(data.values()) or not start_time or not end_time:
            messagebox.showerror("Error", "Name, Specialization, Contact, and Times must be filled.")
            return

        if not available_days:
            messagebox.showerror("Error", "Please select at least one available day.")
            return

        if not validate_time(start_time) or not validate_time(end_time):
            messagebox.showerror("Error", "Time format must be HH:MM (e.g., 10:00 or 17:30).")
            return
            
        name = data["name"]
        if name in DOCTOR_RECORDS:
             messagebox.showerror("Error", f"Doctor ID '{name}' already exists.")
             return

        save_doctor_record(name, {
            "specialization": data["specialization"],
            # Store schedule in the new format
            "schedule": {
                "days": available_days,
                "start_time": start_time,
                "end_time": end_time
            },
            "contact": data["contact"]
        })
        messagebox.showinfo("Success", f"Dr. {name} registered successfully!")
        root.destroy()

    button_style = {'bg': '#00FFCC', 'fg': '#0D1117', 'activebackground': '#00B38F', 'activeforeground': '#0D1117', 'bd': 0, 'font': font_style}
    tk.Button(root, text="Register Doctor", command=submit, **button_style).grid(row=row_idx, column=0, columnspan=2, pady=15, padx=10, sticky="ew")
    
    root.mainloop()

def create_doctor_schedule_view():
    """
    Creates a Tkinter window to display all doctors and their schedules.
    """
    global DOCTOR_RECORDS
    root = tk.Tk()
    root.title(f"Doctor Roster and Schedules ({len(DOCTOR_RECORDS)})")
    root.geometry(TK_WINDOW_GEOMETRY) # Apply standardized geometry
    
    # Define Cyberpunk Style
    BG_COLOR = "#0D1117"
    FG_COLOR = "#00FFCC"
    
    root.configure(bg=BG_COLOR)
    title_font = tkfont.Font(family="Consolas", size=14, weight="bold")
    content_font = tkfont.Font(family="Consolas", size=10)

    # Title
    tk.Label(root, text=":: DOCTOR ROSTER AND SCHEDULES ::", bg=BG_COLOR, fg=FG_COLOR, font=title_font).pack(pady=10, padx=20)
    
    # Scrolled Text Widget for displaying the list
    # Reduced width to fit the 600px geometry
    text_area = scrolledtext.ScrolledText(
        root,
        wrap=tk.WORD,
        width=75, # Adjusted width
        height=25,
        bg="#21262D",
        fg=FG_COLOR,
        insertbackground=FG_COLOR,
        font=content_font,
        bd=0,
        relief=tk.FLAT,
        padx=10,
        pady=10
    )
    text_area.pack(pady=10, padx=10)
    
    # Format and insert doctor data
    if DOCTOR_RECORDS:
        header = f"{'DOCTOR NAME':<15} | {'SPECIALIZATION':<15} | {'HOURS':<11} | AVAILABLE DAYS | CONTACT\n"
        separator = "-" * 75 + "\n"
        text_area.insert(tk.END, header, 'header')
        text_area.insert(tk.END, separator)
        
        # Define a tag for the header
        text_area.tag_config('header', foreground="#FF00CC", font=title_font)
        
        for name, data in sorted(DOCTOR_RECORDS.items()):
            schedule_info = data.get('schedule', {})
            
            # Retrieve the new schedule format
            days = schedule_info.get('days', ['N/A'])
            # Only show first 3 days for compact view
            days_str = ", ".join(days[:3]) + ('...' if len(days) > 3 else '')
            hours_str = f"{schedule_info.get('start_time', 'N/A')}-{schedule_info.get('end_time', 'N/A')}"
                
            line = (
                f"{name:<15} | "
                f"{data['specialization']:<15} | "
                f"{hours_str:<11} | "
                f"{days_str:<15} | "
                f"{data['contact']}\n"
            )
            text_area.insert(tk.END, line)
    else:
        text_area.insert(tk.END, ":: NO DOCTOR RECORDS FOUND ::")
        
    text_area.configure(state='disabled') # Make it read-only
    
    # Close button
    button_style = {'bg': '#00FFCC', 'fg': '#0D1117', 'activebackground': '#00B38F', 'activeforeground': '#0D1117', 'bd': 0, 'font': content_font, 'width': 15}
    tk.Button(root, text="CLOSE", command=root.destroy, **button_style).pack(pady=15)
    
    root.mainloop()

def doctor_management_menu():
    """Displays the menu for doctor management actions."""
    root = tk.Tk()
    root.title("ITS System: Doctor Management")
    root.geometry("500x400") # Smaller geometry for this menu
    
    # Define Cyberpunk Style
    BG_COLOR = "#0D1117"
    FG_COLOR = "#00FFCC"
    ACTIVE_COLOR = "#00B38F"
    
    root.configure(bg=BG_COLOR)
    large_font = tkfont.Font(family="Consolas", size=12, weight="bold")
    
    # Title
    tk.Label(root, text=":: DOCTOR MANAGEMENT CONSOLE ::", bg=BG_COLOR, fg=FG_COLOR, font=large_font).pack(pady=20, padx=50)

    button_style = {
        'bg': FG_COLOR, 'fg': BG_COLOR, 
        'activebackground': ACTIVE_COLOR, 'activeforeground': BG_COLOR, 
        'bd': 0, 'font': large_font, 'width': 40, 'height': 2
    }

    # Commands
    def start_doctor_registration():
        root.destroy()
        create_doctor_registration_form()
        doctor_management_menu()

    def start_doctor_schedule_view():
        root.destroy()
        create_doctor_schedule_view()
        doctor_management_menu()

    # Buttons
    tk.Button(root, text="Register New Doctor", command=start_doctor_registration, **button_style).pack(pady=10)
    tk.Button(root, text="View All Doctor Information/Schedule", command=start_doctor_schedule_view, **button_style).pack(pady=10)
    tk.Button(root, text="<< Back to Main Menu", command=lambda: (root.destroy(), main_menu()), **button_style).pack(pady=20)

    root.mainloop()


def main_menu():
    """Displays the initial menu with three options."""
    root = tk.Tk()
    root.title("ITS System: Main Menu")
    root.geometry(TK_WINDOW_GEOMETRY) # Apply standardized geometry
    
    # Define Cyberpunk Style
    BG_COLOR = "#0D1117"
    FG_COLOR = "#00FFCC"
    ACTIVE_COLOR = "#00B38F"
    
    root.configure(bg=BG_COLOR)
    large_font = tkfont.Font(family="Consolas", size=14, weight="bold")
    
    # Title
    tk.Label(root, text=":: ITS SYSTEM INITIALIZER ::", bg=BG_COLOR, fg=FG_COLOR, font=large_font).pack(pady=20, padx=50)
    
    button_style = {
        'bg': FG_COLOR, 'fg': BG_COLOR, 
        'activebackground': ACTIVE_COLOR, 'activeforeground': BG_COLOR, 
        'bd': 0, 'font': large_font, 'width': 30, 'height': 2
    }

    # Commands
    def start_recognition():
        root.destroy()
        face_recognition_loop()

    def start_registration_flow():
        root.destroy()
        registration_only_flow()

    def start_appointment_flow():
        root.destroy()
        create_appointment_form()
        main_menu() 

    def start_appointment_manage(): # Renamed command for clarity
        root.destroy()
        manage_appointments_window() # Calls the new combined view/cancel window
        main_menu() 
        
    def start_doctor_management(): 
        root.destroy()
        doctor_management_menu()

    # Buttons
    tk.Button(root, text="Existing User (Start Triage)", command=start_recognition, **button_style).pack(pady=10)
    tk.Button(root, text="Register New Patient", command=start_registration_flow, **button_style).pack(pady=10)
    tk.Button(root, text="Appointment Booking", command=start_appointment_flow, **button_style).pack(pady=10)
    tk.Button(root, text="Manage Appointments", command=start_appointment_manage, **button_style).pack(pady=10) # RENAME APPLIED
    tk.Button(root, text="Doctor Management", command=start_doctor_management, **button_style).pack(pady=10)
    tk.Button(root, text="Exit Application", command=root.quit, **button_style).pack(pady=20)

    root.mainloop()

# ===============================
# 4. NEW USER REGISTRATION PROCESS
# ===============================
# ... (Registration functions remain unchanged)

def register_new_user_process(cap, patient_data):
    """
    Captures multiple images for a new user, encodes them, and saves the data.
    """
    global KNOWN_ENCODINGS, KNOWN_NAMES

    new_name = patient_data["name"]
    name_folder = os.path.join(KNOWN_FACES_DIR, new_name)
    os.makedirs(name_folder, exist_ok=True)
    
    new_encodings = []
    
    for i in range(NUM_REGISTRATION_IMAGES):
        message = f"Capturing image {i + 1}/{NUM_REGISTRATION_IMAGES} for {new_name}. Please look at the camera."
        cyber_log(message, "INFO")
        
        # Display feedback in OpenCV window
        ret, frame = cap.read()
        if frame is None or not ret:
            cyber_log("Could not read frame from camera.", "WARNING")
            continue
        
        display_frame = frame.copy()
        H, W, _ = display_frame.shape
        cv2.putText(display_frame, message, (10, H - 10), FONT, 0.7, (255, 255, 0), 2)
        cv2.imshow(f"ITS Registration Monitor - {new_name}", display_frame) 
        cv2.waitKey(2000) # Wait 2 seconds for subject to prepare

        # Processing the captured image
        ret, frame = cap.read()
        if not ret:
            continue
        
        rgb_small_frame = cv2.cvtColor(cv2.resize(frame, (0, 0), fx=FRAME_RESIZE_SCALE, fy=FRAME_RESIZE_SCALE), cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame, model=MODEL)
        
        if len(face_locations) == 1:
            face_encoding = face_recognition.face_encodings(rgb_small_frame, face_locations)[0]
            new_encodings.append(face_encoding)
            
            img_path = os.path.join(name_folder, f"{new_name}_{i+1}.jpg")
            cv2.imwrite(img_path, frame)
            cyber_log(f"Image and encoding captured and saved: {img_path}", "SUCCESS")
        else:
            cyber_log("Found 0 or multiple faces. Retaking this capture...", "WARNING")
            # To ensure we get NUM_REGISTRATION_IMAGES successful captures, decrement i
            i -= 1 
            # Force another loop iteration if capture failed
            continue

    cv2.destroyWindow(f"ITS Registration Monitor - {new_name}")

    if not new_encodings:
        messagebox.showerror("Error", "Failed to capture sufficient images for registration.")
        return 

    # Calculate average encoding from successful captures
    avg_encoding = np.mean(new_encodings, axis=0)
    
    # Update global lists
    KNOWN_ENCODINGS.append(avg_encoding)
    KNOWN_NAMES.append(new_name)

    # Save to disk
    save_encodings(KNOWN_ENCODINGS, KNOWN_NAMES)
    save_patient_record(new_name, patient_data["details"])

    messagebox.showinfo("Success", f"Registration complete for {new_name}.")

def registration_only_flow():
    """Manages the full registration workflow: Form -> Camera -> Save -> Relaunch Menu."""
    cyber_log("Initiating manual registration flow...", "INFO")
    
    # 1. Pop up the GUI form to collect data
    patient_data_from_form = create_registration_form() 
    
    if patient_data_from_form:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cyber_log("Cannot open webcam for registration.", "CRITICAL")
            messagebox.showerror("Error", "Could not access the webcam.")
            # Relaunch main menu on failure
            main_menu() 
            return

        try:
            # 2. Run the capture and saving process
            register_new_user_process(cap, patient_data_from_form)
        finally:
            cap.release()
            
    # Relaunch main menu after process is done (or if user canceled the form)
    main_menu()
    
# ===============================
# 5. CORE FACE RECOGNITION LOOP (CV2 Console UI)
# ===============================
def face_recognition_loop():
    """
    Main CV2 loop for real-time face recognition and triage.
    """
    global KNOWN_ENCODINGS, KNOWN_NAMES, APPOINTMENTS
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cyber_log("Cannot open webcam.", "CRITICAL")
        messagebox.showerror("Error", "Could not access the webcam.")
        return

    cyber_log("Face Recognition Monitor started. Press 'q' to quit.", "INFO")

    # Define BGR colors for CV2 display
    NEON_CYAN = (255, 255, 0) 
    NEON_GREEN = (0, 255, 0)
    CRITICAL_RED_BGR = (0, 0, 255)
    YELLOW_TEXT = (0, 255, 255)
    GRAY_TEXT = (200, 200, 200)
    APPOINTMENT_COLOR = (255, 100, 0) # Orange/Blue for appointments

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame for faster processing
        rgb_small_frame = cv2.cvtColor(cv2.resize(frame, (0, 0), fx=FRAME_RESIZE_SCALE, fy=FRAME_RESIZE_SCALE), cv2.COLOR_BGR2RGB)
        
        # Find all faces and their encodings
        face_locations = face_recognition.face_locations(rgb_small_frame, model=MODEL)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # --- Recognition Logic & Data Retrieval ---
        patient_name_to_display = "Unknown"
        current_details = {}

        for face_encoding, face_loc in zip(face_encodings, face_locations):
            name = "Unknown"
            
            if KNOWN_ENCODINGS:
                matches = face_recognition.compare_faces(KNOWN_ENCODINGS, face_encoding, TOLERANCE)
                if True in matches:
                    match_index = np.argmin(face_recognition.face_distance(KNOWN_ENCODINGS, face_encoding))
                    name = KNOWN_NAMES[match_index]
            
            # The last recognized face's data will populate the panel
            patient_name_to_display = name

            # Scale face locations back to original frame size
            top, right, bottom, left = [v * int(1 / FRAME_RESIZE_SCALE) for v in face_loc]

            # Draw box and label
            color = NEON_GREEN if name != "Unknown" else CRITICAL_RED_BGR
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name.upper(), (left + 6, bottom - 6), FONT, 0.6, (255, 255, 255), 1)


        # --- Determine Triage Panel Data ---
        if patient_name_to_display != "Unknown" and patient_name_to_display in PATIENT_RECORDS:
            current_details = PATIENT_RECORDS[patient_name_to_display]
        else:
            current_details = {
                "age": "N/A", "gender": "N/A", 
                "allergies": "Unknown", "history": "No record found.", 
                "last_visit": "N/A"
            }

        # --- Find Appointments for the Recognized Patient ---
        current_appointments = [
            app for app in APPOINTMENTS 
            if app['name'].lower() == patient_name_to_display.lower()
        ]
        

        # --- Create and Draw Info Panel (Cyberpunk Aesthetic) ---
        H, W, _ = frame.shape
        # Create a solid black info panel
        info_panel = np.zeros((H, INFO_PANEL_WIDTH, 3), dtype=np.uint8)
        
        panel_x = 20
        panel_y = 40
        line_height = 30
        
        # Draw Border and Title
        cv2.line(info_panel, (0, 0), (INFO_PANEL_WIDTH, 0), NEON_CYAN, 3)
        cv2.line(info_panel, (0, H-1), (INFO_PANEL_WIDTH, H-1), NEON_CYAN, 3)
        
        cv2.putText(info_panel, ":: ITS TRIAGE CONSOLE ::", (panel_x, panel_y), FONT, 0.7, NEON_CYAN, 2)
        panel_y += line_height + 5
        cv2.line(info_panel, (panel_x, panel_y - 5), (INFO_PANEL_WIDTH - panel_x, panel_y - 5), (50, 50, 50), 1)

        # PATIENT DETAILS SECTION
        name = patient_name_to_display
        color = NEON_CYAN if name != "Unknown" else CRITICAL_RED_BGR
        status_text = "ID: " + name.upper()
        
        cv2.putText(info_panel, status_text, (panel_x, panel_y + 10), FONT, 0.8, color, 2)
        panel_y += line_height * 2

        details = current_details
        
        # AGE, GENDER, LAST VISIT
        fields = [("AGE", details.get("age", "N/A")), ("GENDER", details.get("gender", "N/A")), ("LAST VISIT", details.get("last_visit", "N/A").split(' ')[0])]
        for label, value in fields:
            cv2.putText(info_panel, f"{label}: {str(value).upper()}", (panel_x, panel_y), FONT, 0.6, NEON_GREEN, 1)
            panel_y += line_height

        # ALLERGIES (Critical Alert)
        allergies = details.get("allergies", "None").upper()
        allergy_color = CRITICAL_RED_BGR if allergies not in ["NONE", "N/A"] and name != "Unknown" else NEON_GREEN
        alert_text = f"ALERT: {allergies}"
        
        panel_y += 10
        cv2.putText(info_panel, alert_text, (panel_x, panel_y), FONT, 0.7, allergy_color, 2)
        
        # History/Triage Note
        panel_y += line_height + 10
        cv2.putText(info_panel, ":: HISTORY LOG ::", (panel_x, panel_y), FONT, 0.6, NEON_CYAN, 1)
        panel_y += line_height
        
        history_text = details.get("history", "NO DATA LOGGED.")
        
        # Simple text wrapping for history
        words = history_text.split()
        current_line = ""
        history_line_height = 20
        for word in words:
            text_size, _ = cv2.getTextSize(current_line + " " + word, FONT, 0.5, 1)
            if text_size[0] < (INFO_PANEL_WIDTH - 2*panel_x):
                current_line += " " + word
            else:
                cv2.putText(info_panel, current_line.strip(), (panel_x, panel_y), FONT, 0.5, GRAY_TEXT, 1)
                panel_y += history_line_height
                current_line = word
        cv2.putText(info_panel, current_line.strip(), (panel_x, panel_y), FONT, 0.5, GRAY_TEXT, 1)
        
        # --- APPOINTMENTS SECTION ---
        panel_y += 30 # Spacer
        cv2.putText(info_panel, ":: APPOINTMENT STATUS ::", (panel_x, panel_y), FONT, 0.6, YELLOW_TEXT, 1)
        panel_y += line_height
        
        if current_appointments:
            for appt in current_appointments:
                # Highlight if appointment is today
                try:
                    appt_date = datetime.strptime(appt['date'], "%Y-%m-%d").date()
                    is_today = appt_date == datetime.now().date()
                except:
                    is_today = False
                    
                color = APPOINTMENT_COLOR if is_today else GRAY_TEXT
                appt_prefix = "TODAY " if is_today else ""
                
                doctor_name = appt.get('doctor', 'UNASSIGNED') 
                appt_text = f"{appt_prefix}{appt['time']} | Dr. {doctor_name} | {appt['reason']}"
                
                cv2.putText(info_panel, appt_text, (panel_x, panel_y), FONT, 0.5, color, 1)
                panel_y += history_line_height
        else:
            cv2.putText(info_panel, "NO APPOINTMENT FOUND.", (panel_x, panel_y), FONT, 0.5, GRAY_TEXT, 1)
            panel_y += history_line_height

        # FOOTER
        cv2.putText(info_panel, datetime.now().strftime("%H:%M:%S"), (panel_x, H - 20), FONT, 0.6, GRAY_TEXT, 1)
        cv2.putText(info_panel, "SYSTEM ONLINE", (INFO_PANEL_WIDTH - 150, H - 20), FONT, 0.6, NEON_GREEN, 1)
        
        
        # Combine the webcam feed and the info panel
        combined_frame = np.concatenate((frame, info_panel), axis=1)

        cv2.imshow("ITS Triage Monitor (Press 'Q' to return to Main Menu)", combined_frame)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("q"):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    cyber_log("Recognition monitor closed. Returning to Main Menu.", "INFO")
    main_menu()


# ===============================
# 6. MAIN EXECUTION
# ===============================

if __name__ == "__main__":
    # 1. Ensure required directories exist
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    
    # 2. Load initial data
    load_patient_records()
    load_appointments()
    load_doctor_records() 
    KNOWN_ENCODINGS, KNOWN_NAMES = load_known_faces()
    
    if len(KNOWN_ENCODINGS) == 0:
        cyber_log("No known faces found. Please use the 'Register New Patient' option.", "WARNING")

    # 3. Start the main menu
    main_menu()

    cyber_log("Application fully terminated.", "INFO")