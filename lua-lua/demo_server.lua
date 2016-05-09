require 'parallel'


local demo_server = torch.class('demo_server')

------------
-- Server class
------------

function demo_server:__init()
    -- Load in general functions
funcs = loadfile("model_functions.lua")
funcs()

    parallel.print('Loading data, parameters, model...')
    self.ext = ""
    self.train_data, self.valid_data, self.model, self.criterion, self.opt = main()

    -- Call the setup script to initialize servers
    if self.opt.setup_servers then
        parallel.print('Setting up remote servers')
        os.execute('python server_init.py ')
    end


    -- Determine where the clients will be located
    if self.opt.remote then
        parallel.print('Runnings clients remotely')
        
        -- Open the list of client ip addresses
        local fh,err = io.open("../client_list.txt")
        if err then print("../client_list.txt not found"); return; end

        -- line by line
        while true do
            local line = fh:read()
            if line == nil then break end
            local addr = self.opt.username .. '@' .. line
            addr = string.gsub(addr, "\n", "") -- remove line breaks
            parallel.addremote( {ip=addr, cores=4, lua=self.opt.torch_path, protocol='ssh -ttq -o "StrictHostKeyChecking no" -i ~/.ssh/gcloud-sshkey'})
            print(addr)
        end
    elseif opt.localhost then
        parallel.print('Running clients through localhost')

        parallel.addremote({ip='localhost', cores=4, lua=opt.torch_path, protocol='ssh -o "StrictHostKeyChecking no" -i ~/.ssh/gcloud-sshkey'})
    elseif opt.kevin then        
        parallel.print('Running clients through Kevins computer')

        parallel.addremote({ip='candokevin@10.251.53.101', cores=4, lua='/Users/candokevin/torch/install/bin/th', protocol='ssh -ttq'})
    end

end

function demo_server:setup_servers()
end

function demo_server:fork_and_exec(worker_code)
    parallel.print('Forking ', opt.n_proc, ' processes')
    parallel.sfork(opt.n_proc)
    parallel.print('Forked')

    -- exec worker code in each process
    parallel.children:exec(worker)
    parallel.print('Finished telling workers to execute')
end

function demo_server:run()

    --send the global parameters to the children
    parallel.children:join()
    parallel.print('Sending parameters to children')
    parallel.children:send({cmd = cmd, arg = arg, ext = opt.extension})

    -- Get the responses from the children
    replies = parallel.children:receive()
    parallel.print('Replies from children', replies)

    -- Train the model
    train(model, criterion, train_data, valid_data)
    parallel.print('Finished training the model')

    -- sync/terminate when all workers are done
    parallel.children:join('break')
    parallel.print('All processes terminated')
end


return demo_server
