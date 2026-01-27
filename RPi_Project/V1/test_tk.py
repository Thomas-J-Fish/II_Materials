import tkinter as tk

root = tk.Tk()
root.title("Tkinter test")
root.geometry("300x200")
label = tk.Label(root, text="If you see this, Tkinter works.")
label.pack(expand=True)

root.mainloop()