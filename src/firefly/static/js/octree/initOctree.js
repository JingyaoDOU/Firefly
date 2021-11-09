function pruneOctree(tree, p){
	out = [];
	tree.forEach(function(d){
		//also set these default values
		d.NparticlesToRender = d.Nparticles;
		d.particleSizeScale = 1.;
		d.particleType = p;
		d.inView = true;
		//d.color =  params.particleColors[p]; //need to set this from the Options file
		d.particles = {'Coordinates':[]};
		if (d.Nparticles > 0) out.push(d);
	})

	return out
}

function formatOctreeCSVdata(data){
	var out = {'Coordinates':[]};
	if (data.length > 0){
		var keys = Object.keys(data[0]);
		if (keys.includes('vx') && keys.includes('vy') && keys.includes('vz')) out.Velocities = [];
		var extraKeys = [];
		keys.forEach(function(k){
			if (k != 'x' && k != 'y' && k != 'z' && k != 'vx' && k != 'vy' && k != 'vz') {
				out[k] = [];
				extraKeys.push(k);
			}
		})
		data.forEach(function(d){
			out.Coordinates.push([parseFloat(d.x), parseFloat(d.y), parseFloat(d.z)]);
			if (out.hasOwnProperty('Velocities')) out.Velocities.push([parseFloat(d.vx), parseFloat(d.vy), parseFloat(d.vz)]);
			extraKeys.forEach(function(k){
				out[k].push(parseFloat(d[k]));
			})
		})
	}

	return out;
}
