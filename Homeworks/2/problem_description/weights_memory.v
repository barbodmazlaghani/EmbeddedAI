`timescale 1ns/1ns

module weights_memory (data);
parameter INPUT_NUMBER = 2;
parameter W_WIDTH = 17;
parameter LAYERID = 1;
parameter NODEID = 1;

output reg [W_WIDTH * (INPUT_NUMBER + 1) - 1:0] data;

(*ram_style = "block"*) reg [W_WIDTH-1:0] memory [0:INPUT_NUMBER];

`ifdef SYNTHESIS
	initial memory[0] = 0;
`else
	integer               				data_in_file    ; // file handler
	integer               				scan_in_file    ; // file handler
	reg   signed [W_WIDTH-1:0] 		captured_data_in;
	integer c;
	initial begin
		// load weight files
		data_in_file = $fopen($sformatf("weights/layer_%1d_%1d_w.mem",LAYERID, NODEID), "r");
		//initialize weight
		for(c=0; c <= INPUT_NUMBER; c= c + 1) begin
			$fscanf(data_in_file, "%h/n", captured_data_in); 
			if (!$feof(data_in_file)) begin
				memory[c] = captured_data_in;
			end
		end
	end
`endif


integer i;
always @* begin
	for (i = 0; i <= INPUT_NUMBER; i=i+1) begin
        data[i*W_WIDTH +: W_WIDTH] = memory[i];
    end
end
endmodule
