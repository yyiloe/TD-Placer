#only LUT FF MUX DSP RAMB CARRY 
#FF only get output_net
#BRAM output_net or output_net and input_net
######### if all delete
set TYPE "net_info"
#########



set script_path [file dirname [info script]]
set targetFolderPath "$script_path/../dataset/ori_data/"
set path_arg $script_path/temp_timing_path.txt
set fo [open "${targetFolderPath}/${TYPE}" "a+"]
set allCells [xilinx::designutils::get_leaf_cells *]
set logicCells [get_cells $allCells -filter {REF_NAME =~ "LUT*" || REF_NAME =~ "F*" || REF_NAME =~ "MUX*" || REF_NAME =~ "CAR*" || REF_NAME =~ "DSP*" || REF_NAME =~ "RAMB*"}]
set countCell [llength $logicCells]
set cnt 0
foreach curCell $logicCells {
	set file_pos [tell $fo]
	set buffer ""

	set output_net ""
	set input_net ""
	set logic_delay ""

	if {[catch {
	
	puts "Processed $cnt of $countCell cells."
	report_timing -through [get_cells $curCell ] -max_paths 1 -file $path_arg
	set ref_name [get_property REF_NAME $curCell]
	set type "0"

	# 判断 REF_NAME，F开头为触发器，RAMB开头为BRAM，其他为0
	if {[string match "F*" $ref_name]} {
    	set type "1"
	} elseif {[string match "RAMB*" $ref_name]} {
    	set type "2"
	} else {
    	set type "0"
	}

	exec python $script_path/parse_timing.py $path_arg $curCell $type
	set fh [open $path_arg r]
	set temp_type [gets $fh]
	if {$temp_type eq "-1"} {
		continue
	} elseif {$temp_type eq "1"} {
		set output_net [gets $fh]
	} elseif {$temp_type eq "0"} {
		set output_net [gets $fh]
		set input_net [gets $fh]
		set logic_delay [gets $fh]
	} elseif {$temp_type eq "2"} {
		set output_net [gets $fh]
		if {[gets $fh line] >= 0} {
			set input_net [gets $fh]
			set logic_delay [gets $fh]
		}
	}
	close $fh

	append buffer "curCell=> $curCell\n"
	#start write outNet to net_info
	set outNet [get_nets $output_net]
	append buffer "outNet=> $outNet\n"
	set pinDriverns [get_pins -leaf -of_objects [get_nets $outNet] -filter {DIRECTION == IN}]
	set outNum [llength [get_pins -leaf -of_objects [get_nets $outNet] -filter {DIRECTION == OUT}]]
	if {$outNum == 0} {
        error "outNum = 0"
    } else {
		set pinDriver [get_pins -leaf -of_objects [get_nets $outNet] -filter {DIRECTION == OUT}]
		foreach pinDriven $pinDriverns {
            set Celldrivern [get_cells -of_objects [get_pins $pinDriven]]
            set curCelltype [get_property REF_NAME $Celldrivern]
            set Numdrivern [llength $Celldrivern]
            if {$Numdrivern > 0} {
                set locDriven [get_property LOC [get_cells $Celldrivern]]
                set delay [get_property FAST_MAX [lindex [get_net_delays -of_objects [get_nets $outNet] -to [get_pins $pinDriven]] 0]]
                append buffer "inPin=> $pinDriven locDriven=> $locDriven celltype=> $curCelltype delay=> $delay\n"
            }
            }
		set Celldriver [get_cells -of_objects [get_pins $pinDriver]]
        set locDriver [get_property LOC [get_cells $Celldriver]]
        set curDriverCelltype [get_property REF_NAME $Celldriver]
        append buffer "outPin=> $pinDriver locDriven=> $locDriver celltype=> $curDriverCelltype\n"
        append buffer "fanall=> [get_property FLAT_PIN_COUNT [get_nets $outNet]]\n"
	}

	append buffer "=============================================\n"
	if {$input_net ne ""} {
		#start write input_Net to net_info
		set inNet [get_nets $input_net]
		append buffer "inNet=> $inNet\n"
		set pinDriverns [get_pins -leaf -of_objects [get_nets $inNet] -filter {DIRECTION == IN}]
		set outNum [llength [get_pins -leaf -of_objects [get_nets $inNet] -filter {DIRECTION == OUT}]]
		if {$outNum == 0} {
        	error "outNum = 0"
    	} else {
				set pinDriver [get_pins -leaf -of_objects [get_nets $inNet] -filter {DIRECTION == OUT}]
				foreach pinDriven $pinDriverns {
            	set Celldrivern [get_cells -of_objects [get_pins $pinDriven]]
            	set curCelltype [get_property REF_NAME $Celldrivern]
            	set Numdrivern [llength $Celldrivern]
            	if {$Numdrivern > 0} {
					set locDriven [get_property LOC [get_cells $Celldrivern]]
                	append buffer "inPin=> $pinDriven locDriven=> $locDriven celltype=> $curCelltype \n"
            	}
            	}
				set Celldriver [get_cells -of_objects [get_pins $pinDriver]]
        		set locDriver [get_property LOC [get_cells $Celldriver]]
        		set curDriverCelltype [get_property REF_NAME $Celldriver]
        		append buffer "outPin=> $pinDriver locDriven=> $locDriver celltype=> $curDriverCelltype\n"
        		append buffer "fanall=> [get_property FLAT_PIN_COUNT [get_nets $inNet]]\n"
				set logic_delay [expr {$logic_delay * 1000}]
				append buffer "logicDelay=> $logic_delay \n"
			}
	}
	
	puts $fo $buffer
    flush $fo


	incr cnt
	
	
	} err]} {
		puts "ERROR on cell $curCell: $err (skipping)"
        seek $fo $file_pos  ;
        continue
	}
	

}