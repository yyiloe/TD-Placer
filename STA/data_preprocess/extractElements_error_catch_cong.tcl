set script_path [file dirname [info script]]
set targetFolderPath "$script_path/../dataset/ori_data/"

set fo [open "${targetFolderPath}/net_info_cong" "a+"]

set allCells [xilinx::designutils::get_leaf_cells *]
set logicCells [get_cells $allCells]
set cnt 0
foreach curCell $logicCells {
    set cellPins [get_pins  -leaf -of_objects [get_cells $curCell]  -filter {DIRECTION == OUT}]
    foreach pinDriver $cellPins {
       incr cnt
       if {$cnt == 1 } {
           set cnt 0
           set pinNet [get_nets -of [get_pins $pinDriver ]]
           set tmp_net $pinNet
           lappend allNets $tmp_net
       }
    }
}

set cnt 0
foreach curNet $allNets {
    set file_pos [tell $fo]
    set buffer ""
    if {[catch {
        append buffer "curNet=> $curNet\n"
        incr cnt
        if {($cnt % 100) == 0} {
        puts "Processed $cnt of [llength $allNets] nets."
    }
        set pinDriverns [get_pins -leaf -of_objects [get_nets $curNet] -filter {DIRECTION == IN}]
        set outNum [llength [get_pins -leaf -of_objects [get_nets $curNet] -filter {DIRECTION == OUT}]]
        
        if {$outNum == 0} {
            append buffer "Skipped (no output pins)\n"
        } else {
            set pinDriver [get_pins -leaf -of_objects [get_nets $curNet] -filter {DIRECTION == OUT}]
            
            foreach pinDriven $pinDriverns {
                set Celldrivern [get_cells -of_objects [get_pins $pinDriven]]
                set curCelltype [get_property REF_NAME $Celldrivern]
                set Numdrivern [llength $Celldrivern]
                
                if {$Numdrivern > 0} {
                    set locDriven [get_property LOC [get_cells $Celldrivern]]
                    set delay [get_property FAST_MAX [lindex [get_net_delays -of_objects [get_nets $curNet] -to [get_pins $pinDriven]] 0]]
                    append buffer "inPin=> $pinDriven locDriven=> $locDriven celltype=> $curCelltype delay=> $delay\n"
                }
            }
            
            set Celldriver [get_cells -of_objects [get_pins $pinDriver]]
            set locDriver [get_property LOC [get_cells $Celldriver]]
            set curDriverCelltype [get_property REF_NAME $Celldriver]
            append buffer "outPin=> $pinDriver locDriven=> $locDriver celltype=> $curDriverCelltype\n"
            append buffer "fanall=> [get_property FLAT_PIN_COUNT [get_nets $curNet]]\n"
        }
        puts $fo $buffer
        flush $fo
        
    } err]} {
        puts "ERROR on net $curNet: $err (skipping)"
        seek $fo $file_pos  ;
        continue
    }
}

close $fo
