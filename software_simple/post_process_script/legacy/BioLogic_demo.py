
import easy_biologic as ebl
import easy_biologic.base_programs as ebp



channels = [0]

params = { 
	'voltages':  [ 0, 1 ]* 2,
	'durations': [ 1 ]* 4, 
    'time_interval': 0.1
}

save_path = 'testt.csv'

bl = ebl.BiologicDevice( 'USB0' )
prg = ebp.CA( bl, params, channels = [0])

prg.run()
prg.save_data(save_path, by_channel = by_channel )