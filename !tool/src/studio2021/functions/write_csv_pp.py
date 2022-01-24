def write_csv_pp(filename, filepath, site, size, footprint, height, num_dadus, num_stories, num_adiabatic, 
                inf_rate, orientation, wwr, frame, polyiso_t, cellulose_t, setback, rear_setback, 
                side_setback, structure_setback, eui, carbon):

    fh = open(filepath, 'w') # write but do not overwrite if filename exists

    fh.write("""filename, site, size, footprint, height, num_dadus, num_stories, num_adiabatic, inf_rate, 
            orientation, WWR, frame, polyiso_t, cellulose_t, setback, rear_setback, side_setback, structure_setback, 
            EUI, embodied_carbon\n""") 
    fh.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format(filename, site, size, 
            footprint, height, num_dadus, num_stories, num_adiabatic, inf_rate, orientation, wwr, frame, polyiso_t, cellulose_t, 
            setback, rear_setback, side_setback, structure_setback, eui, carbon))
   
    fh.close()
    return