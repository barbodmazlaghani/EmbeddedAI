`timescale 1ns/1ns
`default_nettype none

module node #(
    parameter LAYERID=1,
    parameter WIDTH = 16,
    parameter FRACTION = 14,
    parameter W_WIDTH = 8,
    parameter INPUT_NUM = 2,
    parameter OUTPUT_WIDTH = 16,
    parameter NODEID=1
) (
    input wire clk,
    input wire in_ready,
    input wire [WIDTH * INPUT_NUM - 1 : 0] x,

    output reg signed [OUTPUT_WIDTH - 1 : 0] out,
    output reg out_ready
);
localparam INPUT_LEN_DEPTH = $clog2(WIDTH * INPUT_NUM);
localparam INPUT_NUM_DEPTH = $clog2(INPUT_NUM);

localparam real W_FRAC = 7;
localparam real FRAC = 14;
localparam real OUT_FRAC = 21;

integer               				data_in_file    ; // file handler
integer               				scan_in_file    ; // file handler
logic   signed [W_WIDTH-1:0] 		captured_data_in;

(*ram_style = "auto"*)
reg signed [W_WIDTH-1 : 0] w_mem [0:INPUT_NUM];

reg signed [WIDTH * (INPUT_NUM + 1) - 1 : 0] xx;


initial begin
	// load weight files
	data_in_file = $fopen($sformatf("weights/layer_%1d_%1d_w.txt",LAYERID, NODEID), "r");

	
    //initialize weight
	for(int i=0; i <= INPUT_NUM; i= i + 1) begin
		$fscanf(data_in_file, "%h/n", captured_data_in); 
		if (!$feof(data_in_file)) begin
		    w_mem[i] = captured_data_in;
		end
    end
end

reg [INPUT_NUM_DEPTH : 0] i;
reg [INPUT_LEN_DEPTH : 0] j;
reg signed [OUTPUT_WIDTH - 1 : 0] sum;
reg signed [OUTPUT_WIDTH - 1 : 0] r;
reg signed [WIDTH - 1 : 0] signed_x;

always @(posedge clk) begin
    if (in_ready) begin
        xx = {1 << FRACTION, x};
        sum = 0;
        for (i = 0; i <= INPUT_NUM; i= i + 1) begin
            j = i * WIDTH;
            signed_x = xx[j+:WIDTH];
            r = w_mem[i] * signed_x;
            sum = r + sum;
        end
        out <= sum;
        out_ready <= 1'b1;
    end
end
endmodule
`default_nettype wire
